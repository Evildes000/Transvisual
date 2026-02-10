import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from simple_transformer import Transformer

from flask import Flask, jsonify, render_template_string
import sys


# ===== 1. 简单的数据集：把一个随机整数序列当作 src，target 就是同一序列 =====
class CopyDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, pad_idx: int = 0, sos_idx: int = 1, eos_idx: int = 2):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成长度 <= seq_len-2 的随机序列（预留 SOS 和 EOS）
        inner_len = self.seq_len - 2
        # it starts from 3 because 0, 1, 2 are reserved for SOS, EOS, and PAD
        tokens = torch.randint(3, self.vocab_size, (inner_len,), dtype=torch.long)

        # src: [SOS] + tokens + [EOS]
        src = torch.empty(self.seq_len, dtype=torch.long)
        src[0] = self.sos_idx
        src[1:-1] = tokens
        src[-1] = self.eos_idx

        # tgt_input: [SOS] + tokens + [PAD...] （喂给 decoder）
        tgt_input = torch.empty(self.seq_len, dtype=torch.long)
        tgt_input[0] = self.sos_idx
        tgt_input[1:1+inner_len] = tokens
        if 1 + inner_len < self.seq_len:
            tgt_input[1+inner_len:] = self.pad_idx

        # tgt_output(label): tokens + [EOS] + [PAD...]
        tgt_output = torch.empty(self.seq_len, dtype=torch.long)
        tgt_output[0:inner_len] = tokens
        tgt_output[inner_len] = self.eos_idx
        if inner_len + 1 < self.seq_len:
            tgt_output[inner_len+1:] = self.pad_idx

        return src, tgt_input, tgt_output


# ===== 2. 训练循环 =====
def train():
    print(torch.version.cuda)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch built with CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_size = 50      # 小词表，方便演示
    seq_len = 20
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    num_samples = 2000
    batch_size = 32
    num_epochs = 10

    dataset = CopyDataset(num_samples, seq_len, vocab_size, pad_idx, sos_idx, eos_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(f"dataloader is: {dataloader}")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for src, tgt_input, tgt_output in dataloader:
            src = src.to(device)           # (B, L)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            logits = model(src, tgt_input)  # (B, L, vocab_size)
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # (B*L, vocab_size)
                tgt_output.view(-1)                # (B*L)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, loss = {avg_loss:.4f}")

    # 训练完做一个简单测试
    model.eval()
    with torch.no_grad():
        src, tgt_input, tgt_output = dataset[0]
        src = src.unsqueeze(0).to(device)          # (1, L)
        tgt_input = tgt_input.unsqueeze(0).to(device)

        logits = model(src, tgt_input)             # (1, L, vocab_size)
        pred = logits.argmax(dim=-1).squeeze(0).cpu()  # (L,)

        print("Source tokens:     ", src.squeeze(0).cpu().tolist())
        print("Target (label):    ", tgt_output.tolist())
        print("Predicted tokens:  ", pred.tolist())


# ===== 3. 可视化：启动一个简单的 Web 服务，展示 Transformer 各层输出 =====
def _tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def _serialize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


app = Flask(__name__)


@app.route("/")
def index():
    # 一个简单的前端页面：点击按钮，请求后端运行一次 transformer，
    # 并提供矩阵网格可视化（二维矩阵 -> 小方格）以及原始 JSON 查看。
    html = """
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8" />
        <title>Transformer 层可视化 Demo</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 0; background: #f5f5f7; }
            header { padding: 16px 24px; background: #111827; color: #f9fafb; }
            main { padding: 16px 24px; }
            button { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; background: #2563eb; color: #fff; font-size: 14px; }
            button:disabled { opacity: 0.6; cursor: default; }
            .section { margin-top: 16px; padding: 12px 16px; background: #ffffff; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
            pre { max-height: 600px; overflow: auto; background: #0b1120; color: #e5e7eb; padding: 12px; border-radius: 6px; font-size: 12px; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e5e7eb; font-size: 12px; margin-right: 4px; }
        </style>
    </head>
    <body>
        <header>
            <h2>Transformer 每一层输出可视化</h2>
            <div style="font-size: 13px; opacity: 0.8;">包含 token 序列、position encoding、encoder/decoder 各层输出等</div>
        </header>
        <main>
            <div class="section">
                <button id="run-btn" onclick="runOnce()">生成样本并运行 Transformer</button>
                <span id="status" style="margin-left: 12px; font-size: 13px; color: #4b5563;"></span>
            </div>

            <div class="section">
                <div style="margin-bottom:6px; font-size:13px; color:#4b5563;">本次运行的模型配置：</div>
                <div style="font-size:12px; color:#111827; display:flex; flex-wrap:wrap; gap:12px;">
                    <div>d_model: <span id="cfg-d-model">-</span></div>
                    <div>seq_len: <span id="cfg-seq-len">-</span></div>
                    <div>batch_size: <span id="cfg-batch-size">-</span></div>
                    <div>encoder_layers: <span id="cfg-enc-layers">-</span></div>
                    <div>decoder_layers: <span id="cfg-dec-layers">-</span></div>
                </div>
            </div>

            <div class="section">
                <div style="display:flex; gap: 16px; flex-wrap: wrap;">
                    <div style="flex: 1 1 320px;">
                        <div style="margin-bottom:4px; font-size:13px; color:#4b5563;">选择要查看的矩阵：</div>
                        <div style="margin-bottom:4px;">
                            <label style="font-size:13px;">类型：</label>
                            <select id="matrix-type" style="padding:4px 8px; font-size:13px;">
                                <optgroup label="Token IDs">
                                    <option value="src_tokens">src_tokens (源 token id)</option>
                                    <option value="tgt_tokens">tgt_tokens (目标 token id)</option>
                                </optgroup>
                                <optgroup label="Mask">
                                    <option value="src_mask">src_mask (encoder padding mask)</option>
                                    <option value="tgt_mask">tgt_mask (decoder padding & causal)</option>
                                </optgroup>
                                <optgroup label="Encoder">
                                    <option value="encoder.src_embed">encoder.src_embed</option>
                                    <option value="encoder.src_after_pos">encoder.src_after_pos</option>
                                    <option value="encoder.encoder_layers.0.self_attn_q">encoder.layer0.self_attn_q</option>
                                    <option value="encoder.encoder_layers.0.self_attn_k">encoder.layer0.self_attn_k</option>
                                    <option value="encoder.encoder_layers.0.self_attn_v">encoder.layer0.self_attn_v</option>
                                    <option value="encoder.encoder_layers.0.output">encoder.layer0.output</option>
                                    <option value="encoder.encoder_layers.1.self_attn_q">encoder.layer1.self_attn_q</option>
                                    <option value="encoder.encoder_layers.1.self_attn_k">encoder.layer1.self_attn_k</option>
                                    <option value="encoder.encoder_layers.1.self_attn_v">encoder.layer1.self_attn_v</option>
                                    <option value="encoder.encoder_layers.1.output">encoder.layer1.output</option>
                                    <option value="encoder.memory">encoder.memory</option>

                                    
                                </optgroup>
                                <optgroup label="Decoder">
                                    <option value="decoder.tgt_embed">decoder.tgt_embed</option>
                                    <option value="decoder.tgt_after_pos">decoder.tgt_after_pos</option>
                                    <option value="decoder.decoder_layers.0.self_attn_q">decoder.layer0.self_attn_q</option>
                                    <option value="decoder.decoder_layers.0.self_attn_k">decoder.layer0.self_attn_k</option>
                                    <option value="decoder.decoder_layers.0.self_attn_v">decoder.layer0.self_attn_v</option>
                                    <option value="decoder.decoder_layers.0.cross_attn_q">decoder.layer0.cross_attn_q</option>
                                    <option value="decoder.decoder_layers.0.cross_attn_k">decoder.layer0.cross_attn_k</option>
                                    <option value="decoder.decoder_layers.0.cross_attn_v">decoder.layer0.cross_attn_v</option>
                                    <option value="decoder.decoder_layers.0.output">decoder.layer0.output</option>
                                    <option value="decoder.decoder_layers.1.self_attn_q">decoder.layer1.self_attn_q</option>
                                    <option value="decoder.decoder_layers.1.self_attn_k">decoder.layer1.self_attn_k</option>
                                    <option value="decoder.decoder_layers.1.self_attn_v">decoder.layer1.self_attn_v</option>
                                    <option value="decoder.decoder_layers.1.cross_attn_q">decoder.layer1.cross_attn_q</option>
                                    <option value="decoder.decoder_layers.1.cross_attn_k">decoder.layer1.cross_attn_k</option>
                                    <option value="decoder.decoder_layers.1.cross_attn_v">decoder.layer1.cross_attn_v</option>
                                    <option value="decoder.decoder_layers.1.output">decoder.layer1.output</option>
                                    <option value="decoder.decoder_out">decoder.decoder_out</option>
                                </optgroup>

                                <optgroup label="Final Output">
                                    <option value="pre_tokens">final output (最终输出)</option>
                                </optgroup>
                            </select>
                        </div>
                        <div style="margin-bottom:4px;">
                            <label style="font-size:13px;">batch 维 index：</label>
                            <input id="batch-idx" type="number" value="0" min="0" style="width:60px; padding:2px 4px; font-size:13px;" />
                        </div>
                        <div style="margin-bottom:4px; font-size:12px; color:#6b7280;">
                            说明：大多数张量形状为 (batch, seq_len, d_model)，这里我们选择一个batch并展示它所对应的二维矩阵<br/>
                    
                        </div>
                        <button onclick="renderSelectedMatrix()" style="margin-top:4px;">渲染矩阵</button>
                    </div>
                    <div style="flex: 2 1 360px; min-width: 320px;">
                        <div style="margin-bottom:4px; font-size:13px; color:#4b5563;">矩阵可视化：</div>
                        <div id="matrix-container" style="overflow:auto; max-height:480px; border:1px solid #e5e7eb; border-radius:6px; padding:8px; background:#f9fafb; font-size:11px;">// 点击上面的按钮生成一次结果，然后在左侧选择要查看的矩阵。</div>
                    </div>
                </div>
            </div>
        </main>
        <script>
        let lastResult = null;

        function getByPath(obj, path) {
            const parts = path.split('.');
            let cur = obj;
            for (const p of parts) {
                if (cur == null) return null;
                if (Array.isArray(cur)) {
                    const idx = parseInt(p, 10);
                    if (Number.isNaN(idx) || idx < 0 || idx >= cur.length) return null;
                    cur = cur[idx];
                } else {
                    cur = cur[p];
                }
            }
            return cur;
        }

        function renderMatrix(matrix) {
            const container = document.getElementById('matrix-container');
            container.innerHTML = '';
            if (!Array.isArray(matrix) || matrix.length === 0) {
                container.textContent = '矩阵为空或格式不正确';
                return;
            }

            // 如果是一维向量，转成 1 x N 的二维矩阵
            if (!Array.isArray(matrix[0])) {
                matrix = [matrix];
            }

            const rows = matrix.length;
            const cols = matrix[0].length;

            const table = document.createElement('div');
            table.style.display = 'grid';
            table.style.gridTemplateColumns = `repeat(${cols}, minmax(24px, 1fr))`;
            table.style.gap = '1px';

            const flat = [];
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    let v = matrix[i][j];
                    if (typeof v === 'boolean') {
                        v = v ? 1 : 0;
                    }
                    if (typeof v === 'number') flat.push(v);
                }
            }
            let minV = 0, maxV = 0;
            if (flat.length > 0) {
                minV = Math.min(...flat);
                maxV = Math.max(...flat);
            }

            function valueToColor(v) {
                // 简单的蓝-白-红配色，用于大致感受数值大小
                if (maxV === minV) {
                    return '#e5e7eb';
                }
                const t = (v - minV) / (maxV - minV); // 0~1
                const r = Math.round(255 * t);
                const b = Math.round(255 * (1 - t));
                const g = 240;
                return `rgb(${r},${g},${b})`;
            }

            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const cell = document.createElement('div');
                    let v = matrix[i][j];
                    if (typeof v === 'boolean') {
                        v = v ? 1 : 0;
                    }
                    cell.textContent = typeof v === 'number' ? v.toFixed(3) : '';
                    cell.style.borderRadius = '2px';
                    cell.style.padding = '2px 4px';
                    cell.style.textAlign = 'center';
                    cell.style.background = typeof v === 'number' ? valueToColor(v) : '#e5e7eb';
                    cell.style.color = '#111827';
                    cell.style.whiteSpace = 'nowrap';
                    table.appendChild(cell);
                }
            }

            const info = document.createElement('div');
            info.style.marginBottom = '4px';
            info.style.fontSize = '11px';
            info.style.color = '#6b7280';
            info.textContent = `shape: (${rows}, ${cols})    min=${minV.toFixed(4)}, max=${maxV.toFixed(4)}`;

            container.appendChild(info);
            container.appendChild(table);
        }

        function renderSelectedMatrix() {
            if (!lastResult) {
                alert('请先点击“生成样本并运行 Transformer”获取一次结果。');
                return;
            }
            const type = document.getElementById('matrix-type').value;
            const bIdx = parseInt(document.getElementById('batch-idx').value, 10) || 0;

            let tensor = getByPath(lastResult, type);
            if (tensor == null) {
                alert('找不到指定路径的数据: ' + type);
                return;
            }

            // 这里认为张量一般为 3 维 / 4 维 / 2 维 / 1 维。
            // - 3 维: [batch, rows, cols]，先取 batch，展示 (rows x cols)
            // - 4 维: [batch, 1, rows, cols]，先取 batch，再取第 0 个 channel，展示 (rows x cols)（适合 mask）
            // - 2 维: 通常为 [batch, seq_len] 的 token id，按 batch 取一行
            // - 1 维: 当作 1 x N 向量展示
            if (Array.isArray(tensor) && Array.isArray(tensor[0]) && Array.isArray(tensor[0][0]) && Array.isArray(tensor[0][0][0])) {
                // 4 维
                if (bIdx < 0 || bIdx >= tensor.length) {
                    alert('batch 维超出范围');
                    return;
                }
                const mat3d = tensor[bIdx]; // (1, rows, cols) 之类
                const mat2d = mat3d[0];
                renderMatrix(mat2d);
            } else if (Array.isArray(tensor) && Array.isArray(tensor[0]) && Array.isArray(tensor[0][0])) {
                // 3 维
                if (bIdx < 0 || bIdx >= tensor.length) {
                    alert('batch 维超出范围');
                    return;
                }
                const mat2d = tensor[bIdx];
                renderMatrix(mat2d);
            } else if (Array.isArray(tensor) && Array.isArray(tensor[0])) {
                // 2 维矩阵。对于 token id (src_tokens / tgt_tokens)，先按 batch 选一行。
                if (type === 'src_tokens' || type === 'tgt_tokens') {
                    if (bIdx < 0 || bIdx >= tensor.length) {
                        alert('batch 维超出范围');
                        return;
                    }
                    const row = tensor[bIdx]; // (seq_len,)
                    renderMatrix([row]); // 1 x seq_len
                } else {
                    renderMatrix(tensor);
                }
            } else {
                // 一维向量
                renderMatrix(tensor);
            }
        }

        async function runOnce() {
            const btn = document.getElementById('run-btn');
            const status = document.getElementById('status');
            btn.disabled = true;
            status.textContent = '运行中...';
            try {
                const resp = await fetch('/api/run');
                const data = await resp.json();
                lastResult = data;

                // 更新模型配置显示
                if (data.config) {
                    document.getElementById('cfg-d-model').textContent = data.config.d_model;
                    document.getElementById('cfg-seq-len').textContent = data.config.seq_len;
                    document.getElementById('cfg-batch-size').textContent = data.config.batch_size;
                    document.getElementById('cfg-enc-layers').textContent = data.config.encoder_layers;
                    document.getElementById('cfg-dec-layers').textContent = data.config.decoder_layers;
                }

                status.textContent = '完成';
            } catch (e) {
                console.error(e);
                status.textContent = '出错';
            } finally {
                btn.disabled = false;
            }
        }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/api/run")
def run_once():
    """
    生成一个样本，调用 Transformer 的 forward_debug，
    返回包含 tokenization、position encoding、encoder/decoder 各层输出的 JSON。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 50
    seq_len = 20
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    # 生成一个与训练时同分布的样本
    dataset = CopyDataset(1, seq_len, vocab_size, pad_idx, sos_idx, eos_idx)
    src, tgt_input, tgt_output = dataset[0]
    src = src.unsqueeze(0).to(device)
    tgt_input = tgt_input.unsqueeze(0).to(device)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_len=seq_len,
        dropout=0.1,
        pad_idx=pad_idx,
    ).to(device)

    model.eval()
    with torch.no_grad():
        debug_dict = model.forward_debug(src, tgt_input)

    # 将所有 tensor 转为 Python list，方便 JSON 序列化
    payload = _serialize(debug_dict)

    # 顺带把原始标签也一并返回
    payload["tgt_output_label"] = tgt_output.tolist()
    # 模型配置也一并返回，方便前端展示
    payload["config"] = {
        "d_model": 128,
        "seq_len": seq_len,
        "batch_size": 1,  # 这里 /api/run 只生成一个样本
        "encoder_layers": 2,
        "decoder_layers": 2,
    }

    return jsonify(payload)


if __name__ == "__main__":
    # python train.py          -> 正常训练
    # python train.py web      -> 启动可视化 Web 服务
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # 默认在 127.0.0.1:5000 提供一个简单的可视化页面
        app.run(host="127.0.0.1", port=5000, debug=False)
    else:
        train()
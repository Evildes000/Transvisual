import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                   # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)     # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)         # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)         # 奇数维
        pe = pe.unsqueeze(0)                                 # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch, seq_len, d_model)
        mask: (batch, 1, 1, seq_len_k) 或 (batch, 1, seq_len_q, seq_len_k)
        """
        B, Lq, _ = q.size()
        _, Lk, _ = k.size()
        _, Lv, _ = v.size()

        q = self.w_q(q)  # (B, Lq, d_model)
        k = self.w_k(k)  # (B, Lk, d_model)
        v = self.w_v(v)  # (B, Lv, d_model)

        # 分成多头 (B, num_heads, L, d_k)
        q = q.view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, Lv, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力得分: (B, num_heads, Lq, Lk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask == 0 的位置设为 -inf
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和: (B, num_heads, Lq, d_k)
        x = torch.matmul(attn, v)
        # 合并多头
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        x = self.w_o(x)
        return x

    def forward_debug(self, q, k, v, mask=None):
        """
        与 forward 计算过程一致，但额外返回 Q/K/VI 的中间表示，便于可视化。

        为了方便前端展示，Q/K/V 使用尚未拆成多头之前的形状 (B, L, d_model)。
        """
        B, Lq, _ = q.size()
        _, Lk, _ = k.size()
        _, Lv, _ = v.size()

        q_lin = self.w_q(q)  # (B, Lq, d_model)
        k_lin = self.w_k(k)  # (B, Lk, d_model)
        v_lin = self.w_v(v)  # (B, Lv, d_model)

        # 分成多头 (B, num_heads, L, d_k)
        qh = q_lin.view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        kh = k_lin.view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        vh = v_lin.view(B, Lv, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力得分: (B, num_heads, Lq, Lk)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和: (B, num_heads, Lq, d_k)
        x = torch.matmul(attn, vh)
        # 合并多头
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        x_out = self.w_o(x)

        cache = {
            "q": q_lin,
            "k": k_lin,
            "v": v_lin,
        }
        return x_out, cache


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, L, d_model) -> (B, L, d_ff) -> (B, L, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 自注意力
        attn_out = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # 前馈
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

    def forward_debug(self, x, src_mask=None):
        """
        与 forward 计算过程一致，但额外返回中间结果，便于可视化。

        返回:
            out: (B, L, d_model)
            cache: dict，包含本层各阶段输出
        """
        # 自注意力
        attn_out, attn_cache = self.self_attn.forward_debug(x, x, x, src_mask)
        x_sa = x + self.dropout(attn_out)
        x_sa_norm = self.norm1(x_sa)

        # 前馈
        ff_out = self.ff(x_sa_norm)
        x_ff = x_sa_norm + self.dropout(ff_out)
        x_out = self.norm2(x_ff)

        cache = {
            "input": x,
            "self_attn_out": attn_out,
            "self_attn_q": attn_cache["q"],
            "self_attn_k": attn_cache["k"],
            "self_attn_v": attn_cache["v"],
            "after_self_attn_add": x_sa,
            "after_self_attn_norm": x_sa_norm,
            "ff_out": ff_out,
            "after_ff_add": x_ff,
            "output": x_out,
        }
        return x_out, cache


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Decoder 自注意力（带因果 mask）
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Encoder-Decoder 交叉注意力
        attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)

        # 前馈
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x

    def forward_debug(self, x, memory, src_mask=None, tgt_mask=None):
        """
        与 forward 计算过程一致，但额外返回中间结果，便于可视化。

        返回:
            out: (B, L, d_model)
            cache: dict，包含本层各阶段输出
        """
        # Decoder 自注意力（带因果 mask）
        self_attn_out, self_cache = self.self_attn.forward_debug(x, x, x, tgt_mask)
        x_sa = x + self.dropout(self_attn_out)
        x_sa_norm = self.norm1(x_sa)

        # Encoder-Decoder 交叉注意力
        cross_attn_out, cross_cache = self.cross_attn.forward_debug(x_sa_norm, memory, memory, src_mask)
        x_ca = x_sa_norm + self.dropout(cross_attn_out)
        x_ca_norm = self.norm2(x_ca)

        # 前馈
        ff_out = self.ff(x_ca_norm)
        x_ff = x_ca_norm + self.dropout(ff_out)
        x_out = self.norm3(x_ff)

        cache = {
            "input": x,
            "self_attn_out": self_attn_out,
            "self_attn_q": self_cache["q"],
            "self_attn_k": self_cache["k"],
            "self_attn_v": self_cache["v"],
            "after_self_attn_add": x_sa,
            "after_self_attn_norm": x_sa_norm,
            "cross_attn_out": cross_attn_out,
            "cross_attn_q": cross_cache["q"],
            "cross_attn_k": cross_cache["k"],
            "cross_attn_v": cross_cache["v"],
            "after_cross_attn_add": x_ca,
            "after_cross_attn_norm": x_ca_norm,
            "ff_out": ff_out,
            "after_ff_add": x_ff,
            "output": x_out,
        }
        return x_out, cache


def generate_subsequent_mask(size: int, device=None):
    """
    生成 decoder 的因果 mask: (1, 1, size, size)
    上三角为 0（不能看到未来），下三角为 1。

    返回布尔类型，方便与 padding mask 做按位与运算。
    """
    mask = torch.tril(
        torch.ones(size, size, device=device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)
    return mask  # (1, 1, size, size), dtype=bool


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # src: (B, L)
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        return mask

    def make_tgt_mask(self, tgt):
        # padding mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        # causal mask
        L = tgt.size(1)
        causal = generate_subsequent_mask(L, device=tgt.device)         # (1, 1, L, L)
        mask = tgt_pad_mask & causal                                    # (B, 1, L, L)
        return mask

    def encode(self, src, src_mask):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x  # memory

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x  # (B, L, d_model)

    def encode_debug(self, src, src_mask):
        """
        与 encode 计算完全一致，但返回各阶段中间结果，方便前端可视化。
        """
        # token embedding
        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        # position encoding
        src_after_pos = self.pos_encoding(src_embed.clone())

        x = src_after_pos
        encoder_layers_info = []
        for layer in self.encoder_layers:
            x, cache = layer.forward_debug(x, src_mask)
            encoder_layers_info.append(cache)

        return x, {
            "src_embed": src_embed,
            "src_after_pos": src_after_pos,
            "encoder_layers": encoder_layers_info,
        }

    def decode_debug(self, tgt, memory, src_mask, tgt_mask):
        """
        与 decode 计算完全一致，但返回各阶段中间结果，方便前端可视化。
        """
        tgt_embed = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_after_pos = self.pos_encoding(tgt_embed.clone())

        x = tgt_after_pos
        decoder_layers_info = []
        for layer in self.decoder_layers:
            x, cache = layer.forward_debug(x, memory, src_mask, tgt_mask)
            decoder_layers_info.append(cache)

        return x, {
            "tgt_embed": tgt_embed,
            "tgt_after_pos": tgt_after_pos,
            "decoder_layers": decoder_layers_info,
        }

    def forward(self, src, tgt):
        """
        src: (B, L)
        tgt: (B, L)  —— 通常是目标序列右移一位后的形式
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        logits = self.generator(out)  # (B, L, vocab_size)
        return logits

    def forward_debug(self, src, tgt):
        """
        与 forward 计算过程一致，但返回所有关键中间结果，供可视化使用。

        返回:
            result: dict，可 JSON 化之前先做 tensor -> list 的转换
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # encoder 部分
        memory, enc_info = self.encode_debug(src, src_mask)

        # decoder 部分
        dec_out, dec_info = self.decode_debug(tgt, memory, src_mask, tgt_mask)

        logits = self.generator(dec_out)
        pred_tokens = logits.argmax(dim=-1)


        result = {
            "src_tokens": src,
            "tgt_tokens": tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "encoder": {
                "memory": memory,
                **enc_info,
            },
            "decoder": {
                "decoder_out": dec_out,
                **dec_info,
            },
            "logits": logits,
            "pre_tokens": pred_tokens,
        }
        return result
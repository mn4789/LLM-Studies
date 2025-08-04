import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 768  # 4096
    n_layers: int = 12  # 32
    n_heads: int = 12  # 32
    n_kv_heads: Optional[int] = 4
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.2


def precompute_freq_cis(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0/(theta ** (torch.arange(0, head_dim, 2)
                 [:(head_dim//2)].float() / head_dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    xq_r, xq_i = xq.reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = freqs_cos[None, :, None, :]
    freqs_sin = freqs_sin[None, :, None, :]

    xq_out_r = xq_r*freqs_cos - xq_i*freqs_sin
    xq_out_i = xq_r*freqs_sin + xq_i*freqs_cos
    xk_out_r = xk_r*freqs_cos - xk_i*freqs_sin
    xk_out_i = xk_r*freqs_sin + xk_i*freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if not None else args.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.head_dim*self.n_heads, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim *
                            self.n_kv_heads, bias=False)
        self.wv = nn.Linear(self.dim, self.head_dim *
                            self.n_kv_heads, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout_p = args.dropout

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        bsz, seqlen, dim = x.shape
        # QKV
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # ROPE - Relative PositionalEmbeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Grouped MultiQuery Attention - Expand out Keys and Values
        xk = xk.repeat_interleave(self.n_rep, dim=-2)
        xv = xv.repeat_interleave(self.n_rep, dim=-2)

        # Making Heads into batch Dimension
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv,
                                                dropout_p=self.dropout_p if self.training else 0.0,
                                                is_causal=True)
        # Restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.resid_dropout(self.wo(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8 / 3)
            hidden_dim = multiple_of * \
                ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(self.w2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of, dropout=args.dropout)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        # self.params = params
        self.token_embedding = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        freqs_cos, freqs_sin = precompute_freq_cis(
            params.dim//params.n_heads, params.max_seq_len)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # Weight tying Scheme
        self.output.weight = self.token_embedding.weight

        # Init Weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/(2 * params.n_layers)**0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor):
        bsz, seqlen = tokens.shape
        tok_emb = self.token_embedding(tokens)
        h = self.dropout(tok_emb)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        out = self.output(h)
        return out


if __name__ == "__main__":

    # Just ensuring that the model loads without any trouble
    config = ModelArgs()
    model = Transformer(config)

    # Finding out total number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")

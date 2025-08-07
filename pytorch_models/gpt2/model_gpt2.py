from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """ Model configuration """
    n_embd = 768  # 384
    head_size = 64  # 64
    num_heads = 12  # 6
    num_layers = 12  # 6
    block_size = 1024  # 128
    vocab_size = 50257  # 65
    dropout = 0.2  # 0.2


# More efficient implementation of Multihead Attention
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(
            config.n_embd, config.head_size*config.num_heads, bias=True)
        self.key = nn.Linear(
            config.n_embd, config.head_size*config.num_heads, bias=True)
        self.value = nn.Linear(
            config.n_embd, config.head_size*config.num_heads, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.config.num_heads,
                               self.config.head_size).transpose(1, 2)  # (B, Nh, T, head_size)
        k = self.key(x).view(B, T, self.config.num_heads, self.config.head_size).transpose(
            1, 2)  # (B, Nh, T, head_size)
        v = self.value(x).view(B, T, self.config.num_heads,
                               self.config.head_size).transpose(1, 2)  # (B, Nh, T, head_size)
        wei = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.config.dropout)  # (B, Nh, T, head_size)
        wei = wei.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.c_proj(wei)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        # Set attribute on the 2nd Linear layer
        self.net[2].NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mha = CausalMultiHeadAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config)
                                    for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight Tying Scheme
        self.lm_head.weight = self.tok_emb.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        positions = torch.arange(T, device=x.device).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)
        h = self.blocks(h)
        h = self.ln_f(h)
        h = self.lm_head(h)
        return h


if __name__ == "__main__":

    # Just ensuring that the model loads without any trouble
    config = GPTConfig()
    model = GPT2Model(config)

    # Finding out total number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")

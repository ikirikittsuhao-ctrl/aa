import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-6
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x.type_as(
            self.weight
        )

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )

def apply_rope(
    q,
    k,
    cos,
    sin
):
    q = (q * cos) + (
        rotate_half(q) * sin
    )

    k = (k * cos) + (
        rotate_half(k) * sin
    )

    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position,
        theta
    ):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32
                ) / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

        self.max_position = max_position

        self.register_buffer(
            "cos_cached",
            None,
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            None,
            persistent=False
        )

    def forward(
        self,
        seq_len,
        device,
        dtype
    ):
        if (
            self.cos_cached is None
            or self.cos_cached.shape[0] < seq_len
            or self.cos_cached.device != device
        ):
            positions = torch.arange(
                seq_len,
                device=device,
                dtype=self.inv_freq.dtype
            )

            freqs = torch.outer(
                positions,
                self.inv_freq
            )

            emb = torch.cat(
                [freqs, freqs],
                dim=-1
            )

            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        cos = self.cos_cached[
            :seq_len
        ].to(dtype)

        sin = self.sin_cached[
            :seq_len
        ].to(dtype)

        return (
            cos.unsqueeze(0).unsqueeze(0),
            sin.unsqueeze(0).unsqueeze(0)
        )

class GQAAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        n_kv_head,
        dropout
    ):
        super().__init__()

        if n_head % n_kv_head != 0:
            raise ValueError(
                "n_head must be divisible by n_kv_head"
            )

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head

        self.q_proj = nn.Linear(
            dim,
            n_head * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            dim,
            n_kv_head * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            dim,
            n_kv_head * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.dropout = dropout

        self.rope = RotaryEmbedding(
            self.head_dim,
            CFG.block_size,
            CFG.rope_theta
        )

    def forward(
        self,
        x
    ):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope(
            T,
            x.device,
            x.dtype
        )

        q, k = apply_rope(
            q,
            k,
            cos,
            sin
        )

        repeat_factor = (
            self.n_head //
            self.n_kv_head
        )

        if repeat_factor > 1:
            k = k.repeat_interleave(
                repeat_factor,
                dim=1
            )

            v = v.repeat_interleave(
                repeat_factor,
                dim=1
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True
        )

        y = y.transpose(
            1,
            2
        ).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        return self.o_proj(y)

class SwiGLU(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()

        self.gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

        self.up = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

        self.down = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )

    def forward(self, x):
        return self.down(
            F.silu(
                self.gate(x)
            ) * self.up(x)
        )

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm(
            CFG.n_embd
        )

        self.attn = GQAAttention(
            CFG.n_embd,
            CFG.n_head,
            CFG.n_kv_head,
            CFG.dropout
        )

        self.norm2 = RMSNorm(
            CFG.n_embd
        )

        self.mlp = SwiGLU(
            CFG.n_embd,
            CFG.intermediate_size
        )

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x

class SenninLLM(nn.Module):
    def __init__(
        self,
        vocab_size=None
    ):
        super().__init__()

        if vocab_size is None:
            vocab_size = CFG.vocab_size

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size,
            CFG.n_embd
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(CFG.n_layer)
            ]
        )

        self.norm = RMSNorm(
            CFG.n_embd
        )

        self.lm_head = nn.Linear(
            CFG.n_embd,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = (
            self.embedding.weight
        )

        self.apply(
            self._init_weights
        )

        for name, param in self.named_parameters():
            if name.endswith(
                "o_proj.weight"
            ) or name.endswith(
                "down.weight"
            ):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(
                        2 * CFG.n_layer
                    )
                )

    def _init_weights(
        self,
        module
    ):
        if isinstance(
            module,
            nn.Linear
        ):
            if module.weight is not self.embedding.weight:
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        targets=None
    ):
        x = self.embedding(
            input_ids
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        eos_id=None
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids[
                :, -CFG.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :, -1, :
            ]

            logits = logits / max(
                temperature,
                1e-5
            )

            if top_k is not None:
                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                threshold = values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True
                )

                probabilities = torch.softmax(
                    sorted_logits,
                    dim=-1
                )

                cumulative = torch.cumsum(
                    probabilities,
                    dim=-1
                )

                remove = (
                    cumulative > top_p
                )

                remove[:, 1:] = remove[
                    :, :-1
                ].clone()

                remove[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    remove,
                    float("-inf")
                )

                logits = torch.full_like(
                    logits,
                    float("-inf")
                )

                logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits
                )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

            if (
                eos_id is not None
                and next_token.item() == eos_id
            ):
                break

        return input_ids

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )

def print_model_info(model):
    total = count_parameters(
        model
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"parameters: {total:,}"
    )

    print(
        f"trainable: {trainable:,}"
    )

    print(
        f"size fp32: {total * 4 / 1024**3:.2f} GB"
    )

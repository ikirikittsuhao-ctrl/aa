from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    project_name: str = "senninLLM"
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    intermediate_size: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

    batch_size: int = 2
    grad_accum_steps: int = 8

    pretrain_steps: int = 10000
    sft_steps: int = 5000

    eval_interval: int = 250
    eval_batches: int = 30
    save_interval: int = 1000

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0

    seed: int = 42

    data_dir: str = "/content/senninLLM/data"
    checkpoint_dir: str = "/content/senninLLM/checkpoints"
    output_dir: str = "/content/senninLLM/outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def amp_dtype(self):
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @property
    def use_amp(self):
        return self.device == "cuda"

CFG = Config()

Path(CFG.data_dir).mkdir(parents=True, exist_ok=True)
Path(CFG.checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("senninLLM")
print("=" * 60)
print(f"device: {CFG.device}")
print(f"vocab_size: {CFG.vocab_size}")
print(f"block_size: {CFG.block_size}")
print(f"layers: {CFG.n_layer}")
print(f"embedding: {CFG.n_embd}")
print(f"heads: {CFG.n_head}")
print(f"kv_heads: {CFG.n_kv_head}")
print("=" * 60)

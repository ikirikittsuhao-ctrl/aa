import os
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG

class TokenDataset(Dataset):
    def __init__(
        self,
        tokens,
        block_size
    ):
        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

        self.block_size = block_size

    def __len__(self):
        return max(
            0,
            len(self.tokens)
            - self.block_size
        )

    def __getitem__(
        self,
        index
    ):
        x = self.tokens[
            index:
            index + self.block_size
        ]

        y = self.tokens[
            index + 1:
            index + 1 + self.block_size
        ]

        return x, y

class SFTDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        block_size
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(
            self.examples
        )

    def __getitem__(
        self,
        index
    ):
        item = self.examples[index]

        prompt = (
            "<|user|>\n"
            + item["user"]
            + "\n"
            "<|assistant|>\n"
        )

        answer = item["assistant"]

        prompt_ids = self.tokenizer.encode(
            prompt
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_eos=True
        )

        ids = (
            prompt_ids
            + answer_ids
        )

        ids = ids[
            :self.block_size
        ]

        input_ids = ids[:-1]

        target_ids = ids[1:]

        prompt_len = max(
            0,
            min(
                len(prompt_ids),
                len(target_ids)
            )
        )

        target_ids = (
            [-100] * prompt_len
            + target_ids[prompt_len:]
        )

        while len(input_ids) < self.block_size - 1:
            input_ids.append(
                self.tokenizer.pad_id
            )

            target_ids.append(
                -100
            )

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            torch.tensor(
                target_ids,
                dtype=torch.long
            )
        )

def get_lr(
    step,
    max_lr,
    min_lr,
    warmup_steps,
    total_steps
):
    if step < warmup_steps:
        return max_lr * (
            step + 1
        ) / max(
            1,
            warmup_steps
        )

    if step >= total_steps:
        return min_lr

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )

    cosine = (
        0.5
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        min_lr
        + (
            max_lr - min_lr
        ) * cosine
    )

def create_optimizer(
    model
):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim >= 2
            and "norm" not in name.lower()
            and "embedding" not in name.lower()
        ):
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": CFG.weight_decay
            },
            {
                "params": no_decay,
                "weight_decay": 0.0
            }
        ],
        lr=CFG.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    return optimizer

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    path
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": CFG.__dict__
        },
        path
    )

def load_checkpoint(
    model,
    optimizer,
    path,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return (
        checkpoint.get(
            "step",
            0
        ),
        checkpoint.get(
            "loss",
            None
        )
    )

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches
):
    model.eval()

    total_loss = 0.0
    count = 0

    for i, (
        x,
        y
    ) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        _, loss = model(
            x,
            y
        )

        total_loss += loss.item()
        count += 1

    model.train()

    if count == 0:
        return float("inf")

    return total_loss / count

def train(
    model,
    train_loader,
    eval_loader,
    optimizer,
    steps,
    checkpoint_prefix,
    start_step=0
):
    device = CFG.device

    model.train()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            CFG.use_amp
            and CFG.amp_dtype
            == torch.float16
        )
    )

    data_iter = iter(
        train_loader
    )

    best_eval = float("inf")

    for step in range(
        start_step,
        steps
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0

        start_time = time.time()

        for _ in range(
            CFG.grad_accum_steps
        ):
            try:
                x, y = next(
                    data_iter
                )
            except StopIteration:
                data_iter = iter(
                    train_loader
                )

                x, y = next(
                    data_iter
                )

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=CFG.amp_dtype,
                enabled=CFG.use_amp
            ):
                _, loss = model(
                    x,
                    y
                )

                loss = (
                    loss
                    / CFG.grad_accum_steps
                )

            if scaler.is_enabled():
                scaler.scale(
                    loss
                ).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(
                optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG.grad_clip
        )

        if scaler.is_enabled():
            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            optimizer.step()

        lr = get_lr(
            step,
            CFG.learning_rate,
            CFG.min_learning_rate,
            CFG.warmup_steps,
            steps
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        if (
            step % 20 == 0
            or step == steps - 1
        ):
            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"step={step + 1}/{steps} "
                f"loss={total_loss:.4f} "
                f"lr={lr:.7f} "
                f"time={elapsed:.2f}s"
            )

        if (
            eval_loader is not None
            and (
                step % CFG.eval_interval == 0
                or step == steps - 1
            )
        ):
            eval_loss = evaluate(
                model,
                eval_loader,
                device,
                CFG.eval_batches
            )

            print(
                f"eval_loss={eval_loss:.4f}"
            )

            if eval_loss < best_eval:
                best_eval = eval_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    eval_loss,
                    checkpoint_prefix
                    + "_best.pt"
                )

        if (
            step % CFG.save_interval == 0
            and step > 0
        ):
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                total_loss,
                checkpoint_prefix
                + f"_{step + 1}.pt"
            )

    return model

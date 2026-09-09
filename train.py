import os
import json
import random

import torch
from torch.utils.data import DataLoader

from config import CFG
from tokenizer import build_tokenizer
from tokenizer import SenninTokenizer
from model import SenninLLM
from model import print_model_info
from trainer import TokenDataset
from trainer import SFTDataset
from trainer import create_optimizer
from trainer import train

def set_seed(seed):
    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

def prepare_demo_data():
    pretrain_path = os.path.join(
        CFG.data_dir,
        "pretrain.txt"
    )

    sft_path = os.path.join(
        CFG.data_dir,
        "sft.jsonl"
    )

    if not os.path.exists(
        pretrain_path
    ):
        texts = [
            "人工知能は大量のデータからパターンを学習し、文章の生成や分類などを行う技術です。",
            "機械学習では、データと目的に応じてモデルを最適化します。",
            "深層学習では、多数の層を持つニューラルネットワークを使用します。",
            "Transformerは自然言語処理で広く利用されているニューラルネットワーク構造です。",
            "Self Attentionによって文章中のトークン同士の関係を計算できます。",
            "日本語の文章を処理するには、日本語を適切に分割できるTokenizerが重要です。",
            "学習データの品質はモデルの性能に大きな影響を与えます。",
            "モデルの性能を高めるためには、大量で多様なデータが必要です。",
            "推論時には入力された文章から次に続くトークンを予測します。",
            "言語モデルは過去のトークンを条件として次のトークンの確率を計算します。"
        ]

        with open(
            pretrain_path,
            "w",
            encoding="utf-8"
        ) as f:
            for _ in range(1000):
                for text in texts:
                    f.write(
                        text
                        + "\n"
                    )

    if not os.path.exists(
        sft_path
    ):
        examples = [
            {
                "user": "こんにちは",
                "assistant": "こんにちは。今日はどうしましたか？"
            },
            {
                "user": "あなたの名前は？",
                "assistant": "私の名前はsenninLLMです。"
            },
            {
                "user": "AIとは何ですか？",
                "assistant": "AIはデータからパターンを学習し、さまざまな処理を行う技術です。"
            },
            {
                "user": "Transformerとは何ですか？",
                "assistant": "TransformerはAttentionを利用して文章中の情報関係を処理するニューラルネットワークです。"
            },
            {
                "user": "機械学習について説明してください。",
                "assistant": "機械学習はデータから規則やパターンを学習して予測や分類を行う方法です。"
            }
        ]

        with open(
            sft_path,
            "w",
            encoding="utf-8"
        ) as f:
            for _ in range(500):
                for item in examples:
                    f.write(
                        json.dumps(
                            item,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

    return (
        pretrain_path,
        sft_path
    )

def read_text(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

def read_jsonl(path):
    items = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            items.append(
                json.loads(line)
            )

    return items

def split_tokens(tokens):
    split = int(
        len(tokens) * 0.95
    )

    train_tokens = tokens[
        :split
    ]

    eval_tokens = tokens[
        split:
    ]

    return (
        train_tokens,
        eval_tokens
    )

def make_loader(
    dataset,
    shuffle=True
):
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2,
        pin_memory=(
            CFG.device == "cuda"
        )
    )

def train_pretrain(
    model,
    tokenizer,
    pretrain_path
):
    print(
        "\n=== PRETRAIN ==="
    )

    text = read_text(
        pretrain_path
    )

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True
    )

    print(
        f"tokens: {len(tokens):,}"
    )

    train_tokens, eval_tokens = (
        split_tokens(tokens)
    )

    train_dataset = TokenDataset(
        train_tokens,
        CFG.block_size
    )

    eval_dataset = TokenDataset(
        eval_tokens,
        CFG.block_size
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True
    )

    eval_loader = make_loader(
        eval_dataset,
        shuffle=False
    )

    optimizer = create_optimizer(
        model
    )

    train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        CFG.pretrain_steps,
        os.path.join(
            CFG.checkpoint_dir,
            "pretrain"
        )
    )

def train_sft(
    model,
    tokenizer,
    sft_path
):
    print(
        "\n=== SFT ==="
    )

    examples = read_jsonl(
        sft_path
    )

    random.shuffle(
        examples
    )

    split = int(
        len(examples) * 0.9
    )

    train_examples = (
        examples[:split]
    )

    eval_examples = (
        examples[split:]
    )

    train_dataset = SFTDataset(
        train_examples,
        tokenizer,
        CFG.block_size
    )

    eval_dataset = SFTDataset(
        eval_examples,
        tokenizer,
        CFG.block_size
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True
    )

    eval_loader = make_loader(
        eval_dataset,
        shuffle=False
    )

    optimizer = create_optimizer(
        model
    )

    train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        CFG.sft_steps,
        os.path.join(
            CFG.checkpoint_dir,
            "sft"
        )
    )

def chat(
    model,
    tokenizer
):
    print(
        "\n=== CHAT ==="
    )

    model.eval()

    while True:
        try:
            user = input(
                "\nYou: "
            )
        except EOFError:
            break

        if user.strip().lower() in {
            "exit",
            "quit"
        }:
            break

        prompt = (
            "<|user|>\n"
            + user
            + "\n"
            "<|assistant|>\n"
        )

        ids = tokenizer.encode(
            prompt
        )

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device=CFG.device
        )

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                eos_id=tokenizer.eos_id
            )

        generated = output[
            0
        ].tolist()

        text = tokenizer.decode(
            generated
        )

        if "<|assistant|>" in text:
            text = text.split(
                "<|assistant|>",
                1
            )[1]

        if "<|end|>" in text:
            text = text.split(
                "<|end|>",
                1
            )[0]

        if "<|user|>" in text:
            text = text.split(
                "<|user|>",
                1
            )[0]

        print(
            "senninLLM:",
            text.strip()
        )

def main():
    set_seed(
        CFG.seed
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    pretrain_path, sft_path = (
        prepare_demo_data()
    )

    tokenizer_dir = os.path.join(
        CFG.output_dir,
        "tokenizer"
    )

    tokenizer_model = os.path.join(
        tokenizer_dir,
        "sennin_tokenizer.model"
    )

    if not os.path.exists(
        tokenizer_model
    ):
        print(
            "\n=== TRAIN TOKENIZER ==="
        )

        tokenizer = build_tokenizer(
            pretrain_path,
            tokenizer_dir
        )
    else:
        tokenizer = (
            SenninTokenizer(
                tokenizer_model
            )
        )

    print(
        "Tokenizer vocab:",
        tokenizer.vocab_size
    )

    model = SenninLLM(
        tokenizer.vocab_size
    )

    model = model.to(
        CFG.device
    )

    if CFG.device == "cuda":
        model = model.to(
            memory_format=torch.contiguous_format
        )

    print_model_info(
        model
    )

    train_pretrain(
        model,
        tokenizer,
        pretrain_path
    )

    train_sft(
        model,
        tokenizer,
        sft_path
    )

    final_path = os.path.join(
        CFG.checkpoint_dir,
        "senninLLM_final.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": tokenizer.vocab_size
        },
        final_path
    )

    print(
        "\nSaved:",
        final_path
    )

    chat(
        model,
        tokenizer
    )

if __name__ == "__main__":
    main()

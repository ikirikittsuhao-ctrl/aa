import os
import json
import sentencepiece as spm
from config import CFG

class SenninTokenizer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.sp = None

        if model_path is not None and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self):
        if self.sp is None:
            return CFG.vocab_size
        return self.sp.vocab_size()

    @property
    def pad_id(self):
        if self.sp is None:
            return 0
        return self.sp.pad_id()

    @property
    def bos_id(self):
        if self.sp is None:
            return 1
        return self.sp.bos_id()

    @property
    def eos_id(self):
        if self.sp is None:
            return 2
        return self.sp.eos_id()

    @property
    def unk_id(self):
        if self.sp is None:
            return 3
        return self.sp.unk_id()

    def train(self, input_file, model_prefix):
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=CFG.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            normalization_rule_name="nfkc",
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2,
            user_defined_symbols=[
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|end|>"
            ],
            max_sentence_length=16384,
            shuffle_input_sentence=True,
            input_sentence_size=1000000,
            train_extremely_large_corpus=False
        )

        self.model_path = model_prefix + ".model"
        self.sp = spm.SentencePieceProcessor(
            model_file=self.model_path
        )

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):
        if self.sp is None:
            raise RuntimeError(
                "Tokenizer model is not loaded."
            )

        ids = self.sp.encode(
            text,
            out_type=int
        )

        if add_bos:
            ids.insert(0, self.bos_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):
        if self.sp is None:
            raise RuntimeError(
                "Tokenizer model is not loaded."
            )

        return self.sp.decode(ids)

    def save_info(self, path):
        info = {
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                info,
                f,
                ensure_ascii=False,
                indent=2
            )

def build_tokenizer(
    pretrain_file,
    output_dir
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model_prefix = os.path.join(
        output_dir,
        "sennin_tokenizer"
    )

    tokenizer = SenninTokenizer()

    tokenizer.train(
        pretrain_file,
        model_prefix
    )

    tokenizer.save_info(
        os.path.join(
            output_dir,
            "tokenizer.json"
        )
    )

    return tokenizer

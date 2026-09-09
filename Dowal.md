もちろんです。今回の senninLLM 5ファイル版をGoogle Colabでゼロから実行する方法を、初心者向けに順番どおりまとめます。

まず完成形

Colabで最終的にこうなればOKです。

/content/senninLLM/
├── config.py
├── tokenizer.py
├── model.py
├── trainer.py
├── train.py
├── data/
│   ├── pretrain.txt
│   └── sft.jsonl
├── checkpoints/
└── outputs/

data/、checkpoints/、outputs/は最初から作らなくても、train.pyが自動作成します。


---

1. Google Colabを開く

ブラウザでGoogle Colabを開きます。

[Google Colab](https://colab.research.google.com/?utm_source=chatgpt.com)

Googleアカウントでログインします。


---

2. 新しいノートブックを作る

Colabを開いたら、

「ファイル」→「新しいノートブック」

を選びます。

すると、コードを書く場所が表示されます。

例えば、

+ コード
+ テキスト

のようになっています。

ここにPythonコードを入れて実行します。


---

3. GPUを有効にする

これはかなり重要です。

上のメニューから、

ランタイム → ランタイムのタイプを変更

を選択します。

すると設定画面が出ます。

ハードウェア アクセラレータ

なし
↓
T4 GPU

など、利用可能なGPUを選択します。

そして、

保存

を押します。

GPUが利用できるかは、最初に次のコードで確認できます。

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

実行はコード左側の

▶

を押します。

正常なら例えば、

PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA T4

となります。

CUDA available: TrueならGPUが使えています。


---

4. senninLLM用フォルダを作る

次にColabのコードセルで、

!mkdir -p /content/senninLLM

を実行します。

左側のファイルアイコンを押すと、

/content

の中に

senninLLM

が見えるようになります。


---

5. 必要なライブラリをインストール

新しいコードセルを作って、

!pip install -q sentencepiece

を実行します。

PyTorchはColabに基本的に入っているので、今回の構成では追加インストールしなくても動かせます。


---

6. 5つのPythonファイルをアップロード

今回作った、

config.py
tokenizer.py
model.py
trainer.py
train.py

の5ファイルを用意します。

Colab左側の

ファイルアイコン → /content/senninLLM

を開きます。

そこへ5ファイルをアップロードします。

最終的に、

/content/senninLLM/
    config.py
    tokenizer.py
    model.py
    trainer.py
    train.py

となっていることを確認してください。


---

7. ファイルを確認する

Colabで、

!ls -la /content/senninLLM

を実行します。

例えば、

config.py
tokenizer.py
model.py
trainer.py
train.py

が表示されればOKです。


---

8. senninLLMフォルダへ移動

次に、

%cd /content/senninLLM

を実行します。

表示が、

/content/senninLLM

になればOKです。

これは、以降のコマンドをsenninLLMフォルダから実行するためです。


---

9. train.pyを実行

ここが本番です。

!python train.py

を実行します。

すると、

PyTorch: ...
CUDA: True
GPU: NVIDIA T4

などが表示されます。

その後、

=== TRAIN TOKENIZER ===

が表示されます。


---

10. Tokenizerの学習

最初の実行では、

=== TRAIN TOKENIZER ===

が表示されます。

ここでは文章をモデルが扱える「トークン」に分割するためのTokenizerを作ります。

終了すると、

Tokenizer vocab: 32000

のような表示になります。

その後、

parameters: ...
trainable: ...
size fp32: ...

などが表示されます。

これはsenninLLMのモデル情報です。


---

11. Pretrainが始まる

次に、

=== PRETRAIN ===

が表示されます。

ここではモデルに大量の文章を読ませて、

文章の構造
単語の関係
文法
日本語のパターン
次のトークンを予測する能力

などを学習させます。

例えば、

step=21/10000 loss=...
step=41/10000 loss=...
step=61/10000 loss=...

のように進みます。


---

12. lossについて

例えば、

loss=5.2

から、

loss=4.3

loss=3.7

loss=3.1

のように下がっていけば、基本的には学習が進んでいます。

ただし、

lossが下がった＝賢いAIになった

とは限りません。

学習データの量と品質が非常に重要です。


---

13. checkpointが保存される

学習中には、

/content/senninLLM/checkpoints/

にモデルの途中経過が保存されます。

例えば、

pretrain_1000.pt
pretrain_2000.pt
pretrain_best.pt

などです。

万が一Colabのセッションが切れても、保存済みcheckpointがあれば途中から再開できるように発展させられます。


---

14. SFTが始まる

Pretrainが終了すると、

=== SFT ===

になります。

SFTは簡単に言うと、

「文章を予測するモデル」から「人間の指示に答えるモデル」へ近づける学習

です。

例えば、

ユーザー:
こんにちは

アシスタント:
こんにちは。今日はどうしましたか？

のようなデータを使います。


---

15. SFT終了後

SFTが終わると、

Saved:
/content/senninLLM/checkpoints/senninLLM_final.pt

のように表示されます。

これが最終モデルです。


---

16. 会話モードになる

その後、

=== CHAT ===

You:

と表示されます。

ここから実際にsenninLLMと会話できます。

例えば、

You: こんにちは

と入力します。

すると、

senninLLM: こんにちは。今日はどうしましたか？

という形で回答します。

続けて、

You: AIについて教えて

などと入力できます。


---

17. 会話を終了する

終了したい場合は、

You: exit

と入力します。

または、

You: quit

でも終了できます。


---

18. 学習後に作られるファイル

最終的には、

/content/senninLLM/
│
├── config.py
├── tokenizer.py
├── model.py
├── trainer.py
├── train.py
│
├── data/
│   ├── pretrain.txt
│   └── sft.jsonl
│
├── checkpoints/
│   ├── pretrain_best.pt
│   ├── pretrain_1000.pt
│   ├── ...
│   ├── sft_best.pt
│   ├── sft_1000.pt
│   └── senninLLM_final.pt
│
└── outputs/
    └── tokenizer/
        ├── sennin_tokenizer.model
        ├── sennin_tokenizer.vocab
        └── tokenizer.json

という構成になります。


---

19. Colabで一番簡単な実行順

迷ったら、この順番だけ覚えてください。

① GPU確認

import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

↓

② ライブラリ

!pip install -q sentencepiece

↓

③ フォルダ

!mkdir -p /content/senninLLM

↓

④ 5ファイルをアップロード

config.py
tokenizer.py
model.py
trainer.py
train.py

↓

⑤ 移動

%cd /content/senninLLM

↓

⑥ 実行

!python train.py

↓

⑦ 学習

PRETRAIN
↓
SFT

↓

⑧ 会話

=== CHAT ===

You:

↓

⑨ 終了

exit


---

⚠️ 今回のコードで特に注意するところ

現在の train.py には動作確認用の小さなデモデータを自動生成する処理があります。

つまり、何も変更しなくても動きますが、

本格的なデータ
        ↓
Pretrain
        ↓
本格的なSFTデータ
        ↓
SFT
        ↓
高品質な会話AI

というところまではまだ到達しません。

本当に**「ガチ構成のsenninLLM」**にするなら、次の段階では

大量の日本語学習データ

32K〜64K級Tokenizer

会話履歴を保持するChat機能

Pretrain再開機能

SFT再開機能

validation/perplexity

TensorBoard

Google Driveへの自動checkpoint保存

VRAM使用量対策

gradient checkpointing

Flash/SDPA Attention

学習済みモデルの推論専用モード

Web UI


まで入れると、かなり本格的な学習基盤になります。

特にColabではセッション切断対策としてGoogle Driveへのcheckpoint自動保存を入れるのがおすすめです。

# This script tests various Transformer models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
import re
from typing import Dict, List
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

# !pip install nltk, bert-score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction # type: ignore
from bert_score import score # type: ignore

from models.deep_learning.transformer.transformer import Transformer

# Functions for data preprocessing
def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"([^\w\s])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

# Function to encode and pad sequences
def encode_and_pad(tokens: List[str], 
                   vocab: Dict[str, int],
                   max_len: int) -> List[int]:
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]

        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [vocab["<pad>"]] * (max_len - len(ids))

        ids = [min(i, len(vocab)-1) for i in ids]
        ids = [max(i, 0) for i in ids]
        return ids

# Function to evaluate model predictions
def evaluate(preds: torch.Tensor, 
             test_tgt_seqs: torch.Tensor, 
             tgt_vocab: Dict[str, int]) -> None:
    id2tgt = {i: tok for tok, i in tgt_vocab.items()}
    pad_id = tgt_vocab["<pad>"]
    bos_id = tgt_vocab["<bos>"]
    eos_id = tgt_vocab["<eos>"]
    unk_id = tgt_vocab.get("<unk>", None)
    
    pred_sentences = []
    ref_sentences = []
    
    for pred_ids, tgt_ids in zip(preds, test_tgt_seqs):
        pred_ids = pred_ids.tolist() if isinstance(pred_ids, torch.Tensor) else pred_ids
    
        pred_tokens = [id2tgt[i] for i in pred_ids if i not in (pad_id, bos_id, eos_id, unk_id)]
        tgt_tokens  = [id2tgt[i] for i in tgt_ids if i not in (pad_id, bos_id, eos_id)]
    
        if len(pred_tokens) > 0 and len(tgt_tokens) > 0:
            pred_sentences.append(pred_tokens)
            ref_sentences.append([tgt_tokens])
    
    cc = SmoothingFunction()
    bleu = corpus_bleu(ref_sentences, pred_sentences, smoothing_function=cc.method1)
    print(f"BLEU score = {bleu * 100}")
    
    pred_texts = [" ".join(tokens) for tokens in pred_sentences]
    ref_texts  = [" ".join(ref[0]) for ref in ref_sentences]
    
    _, _, F1 = score(pred_texts, ref_texts, lang="de", model_type="xlm-roberta-large")
    print(f"BERTScore F1: {F1.mean().item()}")

if __name__ == "__main__":
    # === Load Dataset === 
    # Load WMT14 en-de dataset
    # https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
    train_path = r"D:\Project\npmod\datawmt-2014-english-german\wmt14_translate_de-en_train.csv"
    test_path = r"D:\Project\npmod\datawmt-2014-english-german\wmt14_translate_de-en_test.csv"

    # Smaller subset for convenience
    train_df = pd.read_csv(train_path, lineterminator="\n")[:5000]
    test_df = pd.read_csv(test_path, lineterminator="\n")[:500]

    train_df["de"] = train_df["de"].apply(clean_text)
    train_df["en"] = train_df["en"].apply(clean_text)
    test_df["de"] = test_df["de"].apply(clean_text)
    test_df["en"] = test_df["en"].apply(clean_text)

    train_src_tokens = [s.split() for s in train_df["en"].tolist()]
    train_tgt_tokens = [s.split() for s in train_df["de"].tolist()]
    test_src_tokens = [s.split() for s in test_df["en"].tolist()]
    test_tgt_tokens = [s.split() for s in test_df["de"].tolist()]

    src_counter = Counter(tok for sent in train_src_tokens for tok in sent)
    tgt_counter = Counter(tok for sent in train_tgt_tokens for tok in sent)

    src_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for i, (tok, _) in enumerate(src_counter.most_common(), start=4):
        src_vocab[tok] = i

    tgt_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for i, (tok, _) in enumerate(tgt_counter.most_common(), start=4):
        tgt_vocab[tok] = i

    # Config
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    d_model = 256
    nhead = 4
    dim_ff = 512
    num_enc_layers = 2
    num_dec_layers = 2
    max_len = 64 

    def encode_and_pad(tokens, vocab, max_len):
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        ids = [vocab["<bos>"]] + ids + [vocab["<eos>"]]

        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [vocab["<pad>"]] * (max_len - len(ids))

        ids = [min(i, len(vocab)-1) for i in ids]
        ids = [max(i, 0) for i in ids]
        return ids

    train_src_seqs = [encode_and_pad(tok_list, src_vocab, max_len) for tok_list in train_src_tokens]
    train_tgt_seqs = [encode_and_pad(tok_list, tgt_vocab, max_len) for tok_list in train_tgt_tokens]
    test_src_seqs = [encode_and_pad(tok_list, src_vocab, max_len) for tok_list in test_src_tokens]
    test_tgt_seqs = [encode_and_pad(tok_list, tgt_vocab, max_len) for tok_list in test_tgt_tokens]

    dataset = TensorDataset(
        torch.tensor(train_src_seqs, dtype=torch.long),
        torch.tensor(train_tgt_seqs, dtype=torch.long)
        )

    test_tensor = TensorDataset(
        torch.tensor(test_src_seqs, dtype=torch.long),
        torch.tensor(test_tgt_seqs, dtype=torch.long),
    )

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,  
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_tensor,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    # ====================


    # === Test Transformer ===
    models = {
        "Transformer": {
            "pe_type": "sinusoidal",
            "norm_type": "layernorm",
            "attn_type": "scaled_dot",
            "ffn_type": "relu",
            "scheduler_type": "noam"
        },
        "T5": {
            "pe_type": "relative",
            "norm_type": "layernorm",
            "attn_type": "scaled_dot",
            "ffn_type": "geglu",
            "scheduler_type": "inverse_sqrt"
        },
        "Linformer": {
            "pe_type": "learned",
            "norm_type": "layernorm",
            "attn_type": "linformer",
            "ffn_type": "relu",
            "scheduler_type": "noam"
        },
        "Conformer": {
            "pe_type": "relative",
            "norm_type": "layernorm",
            "attn_type": "scaled_dot",
            "ffn_type": "gelu",
            "scheduler_type": "cosine"
        },
        "Performer": {
            "pe_type": "sinusoidal",
            "norm_type": "layernorm",
            "attn_type": "performer",
            "ffn_type": "gelu",
            "scheduler_type": "cosine"
        }
    }

    for name, configs in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_ff,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            max_len=max_len,
            pe_type=configs["pe_type"],
            norm_type=configs["norm_type"],
            attn_type=configs["attn_type"],
            ffn_type=configs["ffn_type"]
        )

        model.fit(
            train_loader=train_loader,
            scheduler_type=configs["scheduler_type"],
            number_of_epochs=20,
            verbose=True
        )

        preds = model.predict(test_loader=test_loader)
        evaluate(preds, test_tgt_seqs, tgt_vocab)

    """
    ==============================================================
    Transformer Result
    ==============================================================
    Epoch [5/20], Loss: 0.0479
    Epoch [10/20], Loss: 0.0423
    Epoch [15/20], Loss: 0.0368
    Epoch [20/20], Loss: 0.0306
    BLEU score = 0.2436
    BERTScore F1 = 0.7827

    ==============================================================
    T5 Result
    ==============================================================
    Epoch [5/20], Loss: 0.0440
    Epoch [10/20], Loss: 0.0421
    Epoch [15/20], Loss: 0.0408
    Epoch [20/20], Loss: 0.0398
    BLEU score = 1.4594
    BERTScore F1 = 0.8478

    ==============================================================
    Linformer Result
    ==============================================================
    Epoch [5/20], Loss: 0.0575
    Epoch [10/20], Loss: 0.0483
    Epoch [15/20], Loss: 0.0466
    Epoch [20/20], Loss: 0.0455
    BLEU score = 0.4439
    BERTScore F1 = 0.7625

    ==============================================================
    Conformer Result
    ==============================================================
    Epoch [5/20], Loss: 0.0477
    Epoch [10/20], Loss: 0.0423
    Epoch [15/20], Loss: 0.0369
    Epoch [20/20], Loss: 0.0304
    BLEU score = 0.6250
    BERTScore F1 = 0.8426

    ==============================================================
    Performer Result
    ==============================================================
    Epoch [5/20], Loss: 0.0485
    Epoch [10/20], Loss: 0.0427
    Epoch [15/20], Loss: 0.0319
    Epoch [20/20], Loss: 0.0209
    BLEU score = 0.2597
    BERTScore F1 = 0.7437
    """
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class PairwiseDataset(Dataset):
    def __init__(self, path: str, max_len: int = 256):
        self.samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["query"], item["pos_text"], item["neg_text"]


def pairwise_loss(model, tokenizer, batch, device, max_len: int):
    queries, pos_texts, neg_texts = batch
    # Guard against any empty strings that can create bad tokenization.
    queries = [q or "" for q in queries]
    pos_texts = [p or "" for p in pos_texts]
    neg_texts = [n or "" for n in neg_texts]
    enc_pos = tokenizer(
        list(queries),
        list(pos_texts),
        padding="max_length",
        truncation="longest_first",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    enc_neg = tokenizer(
        list(queries),
        list(neg_texts),
        padding="max_length",
        truncation="longest_first",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    scores_pos = model(**enc_pos).logits.view(-1)
    scores_neg = model(**enc_neg).logits.view(-1)
    # Clamp to keep logits in a stable numeric range.
    scores_pos = torch.clamp(scores_pos, -20.0, 20.0)
    scores_neg = torch.clamp(scores_neg, -20.0, 20.0)
    scores_pos = torch.nan_to_num(scores_pos, nan=0.0, posinf=0.0, neginf=0.0)
    scores_neg = torch.nan_to_num(scores_neg, nan=0.0, posinf=0.0, neginf=0.0)
    diff = torch.clamp(scores_pos - scores_neg, -20.0, 20.0)
    if diff.numel() == 0:
        return None
    # Stable pairwise logistic loss.
    return F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))


def train(model_name: str, train_path: str, val_path: str, epochs: int = 1, lr: float = 1e-5, batch_size: int = 4, max_len: int = 256, output_dir: str = "models/reranker_crossenc/v0.1.0"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(model_name, max_length=max_len, device=device)
    model = ce.model
    tokenizer = ce.tokenizer

    train_ds = PairwiseDataset(train_path, max_len=max_len)
    val_ds = PairwiseDataset(val_path, max_len=max_len) if Path(val_path).exists() else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = pairwise_loss(model, tokenizer, batch, device, max_len)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(1, steps)

        val_loss = None
        if val_loader:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                vsteps = 0
                for batch in val_loader:
                    l = pairwise_loss(model, tokenizer, batch, device, max_len)
                    if l is None or not torch.isfinite(l):
                        continue
                    vloss += l.item()
                    vsteps += 1
                val_loss = vloss / max(1, vsteps)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}" if val_loss is not None else f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ce.save(output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ce.save(output_dir)
    with open(Path(output_dir) / "train_config.json", "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "train_path": train_path,
                "val_path": val_path,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "max_len": max_len,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker (pairwise).")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--train", default="data/reranker/pairwise_train.jsonl")
    parser.add_argument("--val", default="data/reranker/pairwise_val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--output-dir", default="models/reranker_crossenc/v0.1.0")
    args = parser.parse_args()

    train(
        model_name=args.model,
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        output_dir=args.output_dir,
    )

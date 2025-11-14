# src/llm_opt/train_peft.py
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.config import load_config
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

random.seed(42)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

DATA_PATH = Path("results/summaries.json")
OUT_DIR = Path("experiments/peft_flan_t5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class SummDataset(Dataset):
    def __init__(self, items, tok, max_len=512, tgt_len=128):
        self.items, self.tok, self.max_len, self.tgt_len = items, tok, max_len, tgt_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src = f"{self.items[i]['title']}. {self.items[i]['summary']}"
        tgt = self.items[i]["generated"]
        enc = self.tok(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        dec = self.tok(
            tgt,
            truncation=True,
            padding="max_length",
            max_length=self.tgt_len,
            return_tensors="pt",
        )

        # squeeze encoder to (L,)
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        # squeeze labels to (L,) so the batch becomes (B, L)
        labels = dec["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100

        return {**enc, "labels": labels}


def main():
    cfg = load_config()
    base = cfg["summarization"]["models"][0]
    if not DATA_PATH.exists():
        raise SystemExit("Need results/summaries.json from literature pipeline.")
    data = json.loads(DATA_PATH.read_text())
    if len(data) < 8:
        raise SystemExit(
            "Need at least ~8 examples; rerun with larger --max in fetch_papers."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v", "k", "o"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, peft_cfg).to(device)
    model.print_trainable_parameters()

    # simple split
    random.shuffle(data)
    n_train = int(0.8 * len(data))
    train_ds = SummDataset(data[:n_train], tok)
    val_ds = SummDataset(data[n_train:], tok)
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4)

    opt = AdamW(model.parameters(), lr=2e-4)
    steps_total = len(train_dl) * 3
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps_total), steps_total)

    model.train()
    global_step = 0
    best_loss = 9e9
    for epoch in range(3):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            # safety: enforce dtype/keys
            batch["labels"] = batch["labels"].long()
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            global_step += 1

        # quick val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_loss += float(model(**batch).loss)
        val_loss /= max(1, len(val_dl))
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_dir = OUT_DIR / "checkpoint-best"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
        model.train()

    print(f"DONE Best checkpoint -> {OUT_DIR/'checkpoint-best'}")


if __name__ == "__main__":
    main()

"""
Fully-local multi-task GLUE runner for the gated-recurrent TinyBERT student.

Everything is offline: local_files_only=True everywhere, no downloads.

Layout assumed (relative to this script):
    ./local_tinybert/                         # TinyBERT weights + tokenizer
    ./TestFolder/glue_data/{mrpc,stsb,qnli,rte}/{train,dev}.tsv
    ./tinybert_recurrent.py                    # the architecture file

Run:
    python glue_runner_tinybert.py             # all four tasks
    python glue_runner_tinybert.py mrpc rte    # subset
"""

import os
import sys
import csv
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

# Force offline mode at the library level as a hard guarantee — even if some
# code path forgets local_files_only, transformers will refuse to hit the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from tinybert_recurrent import build_student_for_classification

LOCAL_MODEL = "./local_tinybert"
GLUE_DIR    = "./TestFolder/glue_data"
MAX_LEN     = 128
SEED        = 42
torch.manual_seed(SEED)


TASKS = {
    "mrpc": {
        "text_a": 3, "text_b": 4, "label": 0,
        "num_labels": 2, "regression": False,
        "epochs": 5, "lr": 5e-5, "batch": 16, "metric": "acc_f1",
    },
    "stsb": {
        "text_a": 7, "text_b": 8, "label": 9,
        "num_labels": 1, "regression": True,
        "epochs": 5, "lr": 5e-5, "batch": 16, "metric": "pearson_spearman",
    },
    "qnli": {
        "text_a": 1, "text_b": 2, "label": 3,
        "num_labels": 2, "regression": False,
        "epochs": 3, "lr": 5e-5, "batch": 32, "metric": "acc",
        "label_map": {"entailment": 0, "not_entailment": 1},
    },
    "rte": {
        "text_a": 1, "text_b": 2, "label": 3,
        "num_labels": 2, "regression": False,
        "epochs": 5, "lr": 3e-5, "batch": 16, "metric": "acc",
        "label_map": {"entailment": 0, "not_entailment": 1},
    },
}
# Note: TinyBERT is small and benefits from slightly higher LR than DistilBERT;
# 5e-5 default, 3e-5 for the tiny RTE set to avoid thrashing.


class GlueDataset(Dataset):
    def __init__(self, path, cfg, tokenizer):
        self.rows, self.cfg, self.tok = [], cfg, tokenizer
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # header
            for row in reader:
                need = max(cfg["text_a"], cfg["text_b"] or 0, cfg["label"])
                if len(row) <= need:
                    continue
                a = row[cfg["text_a"]].strip()
                b = row[cfg["text_b"]].strip() if cfg["text_b"] is not None else None
                raw = row[cfg["label"]].strip()
                if cfg["regression"]:
                    label = float(raw)
                elif "label_map" in cfg:
                    if raw not in cfg["label_map"]:
                        continue
                    label = cfg["label_map"][raw]
                else:
                    label = int(raw)
                self.rows.append((a, b, label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, label = self.rows[i]
        enc = self.tok(a, b, truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        item["labels"] = (torch.tensor(label, dtype=torch.float)
                          if cfg_regression(self.cfg)
                          else torch.tensor(label, dtype=torch.long))
        return item


def cfg_regression(cfg):
    return cfg["regression"]


# ---- metrics (no scipy dependency) ----
def _f1(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def _pearson(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a-mx)**2 for a in x)); vy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(vx*vy) if vx and vy else 0.0

def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r=[0]*len(v)
        for ri, idx in enumerate(order): r[idx]=ri
        return r
    return _pearson(rank(x), rank(y))


def evaluate(model, loader, cfg, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)   # raw {0,1}; model extends it
            logits = model(input_ids=ids, attention_mask=mask).logits
            if cfg["regression"]:
                preds.extend(logits.squeeze(-1).cpu().tolist())
            else:
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    if cfg["metric"] == "acc":
        return {"acc": sum(p==l for p,l in zip(preds,labels))/len(labels)}
    if cfg["metric"] == "acc_f1":
        return {"acc": sum(p==l for p,l in zip(preds,labels))/len(labels),
                "f1": _f1(preds, labels)}
    return {"pearson": _pearson(preds, labels), "spearman": _spearman(preds, labels)}


def run_task(name, tokenizer, device):
    cfg = TASKS[name]
    tdir = os.path.join(GLUE_DIR, name)
    print(f"\n{'='*60}\nTASK: {name.upper()}\n{'='*60}")

    problem_type = "regression" if cfg["regression"] else None
    model, s_params, t_params = build_student_for_classification(
        LOCAL_MODEL, cfg["num_labels"], problem_type
    )
    model.to(device)
    print(f"Student params: {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")

    train_ds = GlueDataset(os.path.join(tdir, "train.tsv"), cfg, tokenizer)
    dev_ds   = GlueDataset(os.path.join(tdir, "dev.tsv"),   cfg, tokenizer)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")
    if len(train_ds) == 0:
        print(f"!! 0 training rows — check column indices for {name} in TASKS")
        return None

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg["batch"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)

    best = None
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
        metrics = evaluate(model, dev_loader, cfg, device)
        if best is None or list(metrics.values())[0] > list(best.values())[0]:
            best = metrics
        print(f"  epoch {epoch}: " + "  ".join(f"{k}={v:.4f}" for k,v in metrics.items()))

    print(f"BEST {name}: " + "  ".join(f"{k}={v:.4f}" for k,v in best.items()))
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Offline mode:", os.environ.get("TRANSFORMERS_OFFLINE"))

    if not os.path.isdir(LOCAL_MODEL):
        print(f"!! {LOCAL_MODEL} not found"); return
    tokenizer = BertTokenizerFast.from_pretrained(LOCAL_MODEL, local_files_only=True)

    requested = [a.lower() for a in sys.argv[1:]] or list(TASKS.keys())
    results = {}
    for name in requested:
        if name not in TASKS:
            print(f"skip unknown task: {name}"); continue
        r = run_task(name, tokenizer, device)
        if r: results[name] = r

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for name, m in results.items():
        print(f"{name.upper():6s}  " + "  ".join(f"{k}={v:.4f}" for k,v in m.items()))


if __name__ == "__main__":
    main()

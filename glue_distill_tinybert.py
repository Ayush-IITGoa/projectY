"""
End-to-end knowledge distillation for the gated-recurrent TinyBERT student,
teacher fine-tuning included in this one script.

Pipeline (all in one run):
  STAGE 1 - fine-tune a standard 4-layer TinyBERT on the task  -> teacher
  STAGE 2 - distill teacher into the 2-block recurrent student  -> student

Self-distillation: teacher and student both start from the same local TinyBERT;
student is the compressed (4->2 layer, recurrent) version. Same hidden size
(312) both sides, so optional hidden-state matching needs no projection.

Loss = alpha * KD_soft (KL on temperature-scaled logits, x T^2)
     + (1 - alpha) * CE_hard (true labels)
     + beta  * hidden_MSE   (optional; student's 4 effective passes vs teacher's
                             4 layer outputs)

Fully local / offline.

Run:
    python kd_tinybert.py            # default task sst2
    python kd_tinybert.py rte
"""

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tinybert_recurrent import build_student_for_classification

# ---- paths ----
PRETRAINED_TINYBERT = "./local_tinybert"   # base model: seeds teacher AND student

# ---- KD hyperparameters ----
TEMPERATURE = 4.0
ALPHA       = 0.7     # weight on soft KD vs hard CE
BETA        = 0.0     # hidden-state MSE weight; set >0 (e.g. 0.1) to enable
MAX_LEN     = 128
SEED        = 42
torch.manual_seed(SEED)

GLUE_DIR = "./TestFolder/glue_data"

TASKS = {
    "sst2": {"text_a": 0, "text_b": None, "label": 1, "num_labels": 2,
             "teacher_epochs": 3, "student_epochs": 3, "lr": 5e-5, "batch": 32},
    "mrpc": {"text_a": 3, "text_b": 4, "label": 0, "num_labels": 2,
             "teacher_epochs": 5, "student_epochs": 5, "lr": 5e-5, "batch": 16},
    "rte":  {"text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "teacher_epochs": 5, "student_epochs": 5, "lr": 3e-5, "batch": 16,
             "label_map": {"entailment": 0, "not_entailment": 1}},
    "qnli": {"text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "teacher_epochs": 3, "student_epochs": 3, "lr": 5e-5, "batch": 32,
             "label_map": {"entailment": 0, "not_entailment": 1}},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class GlueDataset(Dataset):
    def __init__(self, path, cfg, tokenizer):
        self.rows, self.cfg, self.tok = [], cfg, tokenizer
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)
            for row in reader:
                need = max(cfg["text_a"], cfg["text_b"] or 0, cfg["label"])
                if len(row) <= need:
                    continue
                a = row[cfg["text_a"]].strip()
                b = row[cfg["text_b"]].strip() if cfg["text_b"] is not None else None
                raw = row[cfg["label"]].strip()
                if "label_map" in cfg:
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
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)}


def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            correct += (logits.argmax(-1) == lab).sum().item()
            total += lab.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# STAGE 1 - teacher fine-tuning (standard 4-layer TinyBERT)
# ---------------------------------------------------------------------------
def train_teacher(cfg, train_loader, dev_loader, device):
    print(f"\n{'='*60}\nSTAGE 1: TEACHER FINE-TUNING (standard 4-layer TinyBERT)\n{'='*60}")
    teacher = BertForSequenceClassification.from_pretrained(
        PRETRAINED_TINYBERT, num_labels=cfg["num_labels"], local_files_only=True
    ).to(device)

    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {t_params:,}")

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["teacher_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)

    best = 0.0
    for epoch in range(cfg["teacher_epochs"]):
        teacher.train(); running = 0.0
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            optimizer.zero_grad()
            out = teacher(input_ids=ids, attention_mask=mask, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            running += out.loss.item()
            if step % 100 == 0:
                print(f"  [teacher] epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running/(step+1):.4f}")
        acc = evaluate(teacher, dev_loader, device)
        best = max(best, acc)
        print(f"[teacher] Epoch {epoch}: dev accuracy = {acc:.4f}")
    print(f"[teacher] BEST dev accuracy = {best:.4f}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if BETA > 0:
        teacher.config.output_hidden_states = True
    return teacher, best, t_params


# ---------------------------------------------------------------------------
# STAGE 2 - distillation into recurrent student
# ---------------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1.0 - alpha) * hard, soft.item(), hard.item()


def attach_hidden_capture(student):
    """Monkeypatch blocks to stash pass_a/pass_b (only used when BETA>0)."""
    captures = {}
    def make_hook(idx, block):
        def forward(hidden_states, attention_mask=None, **kwargs):
            pass_a = block.shared_layer(hidden_states, attention_mask)[0]
            pass_b = block.shared_layer(pass_a, attention_mask)[0]
            gate = torch.sigmoid(block.gate(pass_a))
            output = gate * pass_b + (1.0 - gate) * pass_a
            captures[f"{idx}_a"] = pass_a
            captures[f"{idx}_b"] = pass_b
            return (output,)
        block.forward = forward
    for idx, block in enumerate(student.bert.encoder.layer):
        make_hook(idx, block)
    return captures


def hidden_mse(student_hiddens, teacher_hidden_states):
    t = teacher_hidden_states[1:5]   # 4 teacher layer outputs
    loss = 0.0
    for s_h, t_h in zip(student_hiddens, t):
        loss = loss + F.mse_loss(s_h, t_h)
    return loss / len(student_hiddens)


def distill_student(cfg, teacher, t_params, train_loader, dev_loader, device):
    print(f"\n{'='*60}\nSTAGE 2: DISTILLATION (recurrent 2-block student)\n{'='*60}")
    student, s_params, _ = build_student_for_classification(
        PRETRAINED_TINYBERT, cfg["num_labels"]
    )
    student.to(device)
    print(f"Student params: {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")

    captures = attach_hidden_capture(student) if BETA > 0 else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["student_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)

    best = 0.0
    for epoch in range(cfg["student_epochs"]):
        student.train()
        run_soft = run_hard = run_hid = 0.0
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)

            with torch.no_grad():
                t_out = teacher(input_ids=ids, attention_mask=mask)

            s_logits = student(input_ids=ids, attention_mask=mask).logits
            loss, soft_v, hard_v = kd_loss(s_logits, t_out.logits, lab, TEMPERATURE, ALPHA)

            if BETA > 0:
                s_hiddens = [captures["0_a"], captures["0_b"],
                             captures["1_a"], captures["1_b"]]
                h = hidden_mse(s_hiddens, t_out.hidden_states)
                loss = loss + BETA * h
                run_hid += h.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            run_soft += soft_v; run_hard += hard_v
            if step % 100 == 0:
                msg = (f"  [student] epoch {epoch} step {step}/{len(train_loader)} "
                       f"soft={run_soft/(step+1):.4f} hard={run_hard/(step+1):.4f}")
                if BETA > 0:
                    msg += f" hid={run_hid/(step+1):.4f}"
                print(msg)

        acc = evaluate(student, dev_loader, device)
        best = max(best, acc)
        print(f"[student] Epoch {epoch}: dev accuracy = {acc:.4f}")
    print(f"[student] BEST dev accuracy (distilled) = {best:.4f}")
    return student, best, s_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    task = (sys.argv[1].lower() if len(sys.argv) > 1 else "sst2")
    if task not in TASKS:
        print(f"unknown task: {task}; choose from {list(TASKS)}"); return
    cfg = TASKS[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| task:", task)
    print("Offline:", os.environ.get("TRANSFORMERS_OFFLINE"))

    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_TINYBERT, local_files_only=True)
    tdir = os.path.join(GLUE_DIR, task)
    train_ds = GlueDataset(os.path.join(tdir, "train.tsv"), cfg, tokenizer)
    dev_ds   = GlueDataset(os.path.join(tdir, "dev.tsv"),   cfg, tokenizer)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")
    if len(train_ds) == 0:
        print("!! 0 training rows — check column indices for this task"); return
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg["batch"])

    teacher, teacher_acc, t_params = train_teacher(cfg, train_loader, dev_loader, device)
    student, student_acc, s_params = distill_student(
        cfg, teacher, t_params, train_loader, dev_loader, device
    )

    print(f"\n{'='*60}\nSUMMARY ({task})\n{'='*60}")
    print(f"Teacher (4-layer)  dev acc : {teacher_acc:.4f}  params {t_params:,}")
    print(f"Student (2-block)  dev acc : {student_acc:.4f}  params {s_params:,} "
          f"({100*s_params/t_params:.1f}% of teacher)")
    print(f"Accuracy retained          : {100*student_acc/teacher_acc:.1f}%")

    out = f"./distilled_{task}"
    os.makedirs(out, exist_ok=True)
    student.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print("Saved distilled student to", out)


if __name__ == "__main__":
    main()

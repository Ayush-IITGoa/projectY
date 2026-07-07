"""
End-to-end knowledge distillation for the gated-recurrent BERT-base student.
Teacher fine-tuning + distillation in one script.

  STAGE 1 - fine-tune standard 12-layer BERT-base on the task -> teacher
  STAGE 2 - distill teacher into the 6-block recurrent student -> student

Self-distillation: teacher and student both start from local BERT-base-uncased;
student is the compressed (12->6 layer, recurrent) version. Hidden size 768 on
both sides, so optional hidden-state matching needs no projection.

Tasks: sst2, mrpc, cola, qqp, qnli, rte  (COLA has NO header row)

Hardware notes (tuned for 1x 16GB GPU, 16-core CPU):
  - AMP mixed precision (torch.amp) for GPU utilization + memory headroom
  - pre-tokenization once up front (not per __getitem__)
  - TF32 enabled, pinned memory, multi-worker loaders
  - resume at EPOCH granularity for both teacher and student

Outputs:
  checkpoints -> ./checkpoints_bertbase/{task}/teacher , .../student
  best models -> ./best_models_bertbase/{task}/teacher , .../student
  logs        -> ./logs/bertBase/kd_bertbase_{task}_{timestamp}.log

Fully local / offline. Run:
    python kd_bertbase.py cola
    python kd_bertbase.py qqp
"""

import os
import sys
import csv
import time
import json
import logging
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from bertbase_recurrent import build_student_for_classification

# ---- paths ----
PRETRAINED_BERT = "./local_bertbaseuncased"     # seeds teacher AND student
GLUE_DIR        = "./TestFolder/glue_data"
CKPT_ROOT       = "./checkpoints_bertbase"
BEST_ROOT       = "./best_models_bertbase"
LOG_DIR         = "./logs/bertBase"

# ---- KD hyperparameters ----
TEMPERATURE = 4.0
ALPHA       = 0.7
BETA        = 0.0        # hidden-state MSE weight; set >0 (e.g. 0.1) to enable
MAX_LEN     = 128
SEED        = 42
torch.manual_seed(SEED)

# ---- hardware / throughput ----
NUM_WORKERS = 8          # 16-core CPU -> 8 workers is a good balance
USE_AMP     = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# CoLA columns: code \t label \t acceptability \t sentence  (NO header)
# QQP columns : id \t qid1 \t qid2 \t question1 \t question2 \t is_duplicate
# MRPC columns: Quality \t #1 ID \t #2 ID \t #1 String \t #2 String
TASKS = {
    "sst2": {"dir": "sst2", "text_a": 0, "text_b": None, "label": 1, "num_labels": 2,
             "has_header": True, "metric": "acc",
             "teacher_epochs": 3, "student_epochs": 3, "lr": 2e-5, "batch": 32},
    "mrpc": {"dir": "mrpc", "text_a": 3, "text_b": 4, "label": 0, "num_labels": 2,
             "has_header": True, "metric": "acc_f1",
             "teacher_epochs": 5, "student_epochs": 5, "lr": 2e-5, "batch": 32},
    "cola": {"dir": "cola", "text_a": 3, "text_b": None, "label": 1, "num_labels": 2,
             "has_header": False, "metric": "mcc",
             "teacher_epochs": 5, "student_epochs": 5, "lr": 2e-5, "batch": 32},
    "qqp":  {"dir": "qqp_new", "text_a": 3, "text_b": 4, "label": 5, "num_labels": 2,
             "has_header": True, "metric": "acc_f1",
             "teacher_epochs": 3, "student_epochs": 3, "lr": 2e-5, "batch": 64},
    "qnli": {"dir": "qnli", "text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "has_header": True, "metric": "acc",
             "label_map": {"entailment": 0, "not_entailment": 1},
             "teacher_epochs": 3, "student_epochs": 3, "lr": 2e-5, "batch": 32},
    "rte":  {"dir": "rte", "text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "has_header": True, "metric": "acc",
             "label_map": {"entailment": 0, "not_entailment": 1},
             "teacher_epochs": 5, "student_epochs": 5, "lr": 2e-5, "batch": 32},
}


# ---------------------------------------------------------------------------
# Logging: to file AND terminal
# ---------------------------------------------------------------------------
def setup_logger(task):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"kd_bertbase_{task}_{ts}.log")
    logger = logging.getLogger(f"kd_{task}_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(logfile); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"Log file: {logfile}")
    return logger


# ---------------------------------------------------------------------------
# Data — pre-tokenized once up front
# ---------------------------------------------------------------------------
class GlueDataset(Dataset):
    def __init__(self, path, cfg, tokenizer, logger):
        texts_a, texts_b, labels = [], [], []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            if cfg["has_header"]:
                next(reader)
            need = max(cfg["text_a"], cfg["text_b"] or 0, cfg["label"])
            for row in reader:
                if len(row) <= need:
                    continue
                raw = row[cfg["label"]].strip()
                if "label_map" in cfg:
                    if raw not in cfg["label_map"]:
                        continue
                    label = cfg["label_map"][raw]
                else:
                    try:
                        label = int(raw)
                    except ValueError:
                        continue
                texts_a.append(row[cfg["text_a"]].strip())
                texts_b.append(row[cfg["text_b"]].strip() if cfg["text_b"] is not None else None)
                labels.append(label)

        # Tokenize the whole split in one batched call (fast tokenizer, multi-thread)
        t0 = time.time()
        if cfg["text_b"] is None:
            enc = tokenizer(texts_a, truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
        else:
            enc = tokenizer(texts_a, texts_b, truncation=True, max_length=MAX_LEN,
                            padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)
        logger.info(f"  pre-tokenized {len(labels)} rows from {os.path.basename(path)} "
                    f"in {time.time()-t0:.1f}s")

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _f1(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def _mcc(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    num = (tp*tn) - (fp*fn)
    den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return num/den if den else 0.0

def compute_metric(preds, labels, metric):
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    if metric == "acc":
        return {"acc": acc}, acc
    if metric == "acc_f1":
        f1 = _f1(preds, labels)
        return {"acc": acc, "f1": f1}, (acc + f1) / 2   # selection score
    if metric == "mcc":
        mcc = _mcc(preds, labels)
        return {"acc": acc, "mcc": mcc}, mcc            # select on MCC
    return {"acc": acc}, acc


def evaluate(model, loader, device, metric):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=USE_AMP):
                logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return compute_metric(preds, labels, metric)


# ---------------------------------------------------------------------------
# Checkpoint helpers (epoch-level resume)
# ---------------------------------------------------------------------------
def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, best_score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,               # last COMPLETED epoch
        "best_score": best_score,
    }, path)

def load_ckpt(path, model, optimizer, scheduler, scaler, device, logger):
    if not os.path.isfile(path):
        return -1, 0.0                # no checkpoint: start_epoch=0, best=0
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    logger.info(f"  resumed from {path} @ completed epoch {ck['epoch']} "
                f"(best={ck['best_score']:.4f})")
    return ck["epoch"], ck["best_score"]


def save_best_model(kind, task, model, tokenizer, logger):
    out = os.path.join(BEST_ROOT, task, kind)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info(f"  ** new best {kind} saved -> {out}")


# ---------------------------------------------------------------------------
# STAGE 1 - teacher fine-tuning
# ---------------------------------------------------------------------------
def train_teacher(cfg, task, train_loader, dev_loader, device, tokenizer, logger):
    logger.info("="*70)
    logger.info("STAGE 1: TEACHER FINE-TUNING (standard 12-layer BERT-base)")
    logger.info("="*70)
    teacher = BertForSequenceClassification.from_pretrained(
        PRETRAINED_BERT, num_labels=cfg["num_labels"], local_files_only=True
    ).to(device)
    t_params = sum(p.numel() for p in teacher.parameters())
    logger.info(f"Teacher params: {t_params:,}")

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["teacher_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    ckpt = os.path.join(CKPT_ROOT, task, "teacher", "last.pt")
    start_epoch, best = load_ckpt(ckpt, teacher, optimizer, scheduler, scaler, device, logger)
    start_epoch += 1

    for epoch in range(start_epoch, cfg["teacher_epochs"]):
        teacher.train(); running = 0.0; t0 = time.time()
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lab  = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=USE_AMP):
                out = teacher(input_ids=ids, attention_mask=mask, labels=lab)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += out.loss.item()
            if step % 100 == 0:
                seen = step + 1
                rate = (seen * ids.size(0)) / (time.time() - t0)
                logger.info(f"  [teacher] ep {epoch} step {step}/{len(train_loader)} "
                            f"loss {running/seen:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                            f"{rate:.0f} ex/s")
        metrics, score = evaluate(teacher, dev_loader, device, cfg["metric"])
        mstr = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"[teacher] Epoch {epoch}: {mstr}  (epoch time {time.time()-t0:.0f}s)")
        save_ckpt(ckpt, teacher, optimizer, scheduler, scaler, epoch, best)
        if score > best:
            best = score
            save_best_model("teacher", task, teacher, tokenizer, logger)

    logger.info(f"[teacher] BEST selection score = {best:.4f}")

    # Reload best teacher weights for distillation
    best_dir = os.path.join(BEST_ROOT, task, "teacher")
    if os.path.isdir(best_dir):
        teacher = BertForSequenceClassification.from_pretrained(
            best_dir, num_labels=cfg["num_labels"], local_files_only=True
        ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if BETA > 0:
        teacher.config.output_hidden_states = True
    return teacher, best, t_params


# ---------------------------------------------------------------------------
# STAGE 2 - distillation
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
    """Monkeypatch blocks to stash pass_a/pass_b (only used when BETA>0).
    For BERT-base there are 6 blocks -> 12 effective passes."""
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


def hidden_mse(captures, teacher_hidden_states):
    # Student 12 effective passes -> teacher 12 layer outputs (indices 1..12)
    t = teacher_hidden_states[1:13]
    order = []
    for i in range(6):
        order.append(captures[f"{i}_a"]); order.append(captures[f"{i}_b"])
    loss = 0.0
    for s_h, t_h in zip(order, t):
        loss = loss + F.mse_loss(s_h, t_h)
    return loss / len(order)


def distill_student(cfg, task, teacher, t_params, train_loader, dev_loader,
                    device, tokenizer, logger):
    logger.info("="*70)
    logger.info("STAGE 2: DISTILLATION (recurrent 6-block BERT-base student)")
    logger.info("="*70)
    student, s_params, _ = build_student_for_classification(
        PRETRAINED_BERT, cfg["num_labels"]
    )
    student.to(device)
    logger.info(f"Student params: {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")

    captures = attach_hidden_capture(student) if BETA > 0 else None

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["student_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    ckpt = os.path.join(CKPT_ROOT, task, "student", "last.pt")
    start_epoch, best = load_ckpt(ckpt, student, optimizer, scheduler, scaler, device, logger)
    start_epoch += 1

    for epoch in range(start_epoch, cfg["student_epochs"]):
        student.train()
        run_soft = run_hard = run_hid = 0.0; t0 = time.time()
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lab  = batch["labels"].to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast("cuda", enabled=USE_AMP):
                t_out = teacher(input_ids=ids, attention_mask=mask)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=USE_AMP):
                s_logits = student(input_ids=ids, attention_mask=mask).logits
                loss, soft_v, hard_v = kd_loss(s_logits, t_out.logits, lab,
                                               TEMPERATURE, ALPHA)
                if BETA > 0:
                    h = hidden_mse(captures, t_out.hidden_states)
                    loss = loss + BETA * h
                    run_hid += h.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

            run_soft += soft_v; run_hard += hard_v
            if step % 100 == 0:
                seen = step + 1
                rate = (seen * ids.size(0)) / (time.time() - t0)
                msg = (f"  [student] ep {epoch} step {step}/{len(train_loader)} "
                       f"soft={run_soft/seen:.4f} hard={run_hard/seen:.4f}")
                if BETA > 0:
                    msg += f" hid={run_hid/seen:.4f}"
                msg += f" {rate:.0f} ex/s"
                logger.info(msg)

        metrics, score = evaluate(student, dev_loader, device, cfg["metric"])
        mstr = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"[student] Epoch {epoch}: {mstr}  (epoch time {time.time()-t0:.0f}s)")
        save_ckpt(ckpt, student, optimizer, scheduler, scaler, epoch, best)
        if score > best:
            best = score
            save_best_model("student", task, student, tokenizer, logger)

    logger.info(f"[student] BEST selection score (distilled) = {best:.4f}")
    return student, best, s_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    task = (sys.argv[1].lower() if len(sys.argv) > 1 else "sst2")
    if task not in TASKS:
        print(f"unknown task: {task}; choose from {list(TASKS)}"); return
    cfg = TASKS[task]
    logger = setup_logger(task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | task: {task} | AMP: {USE_AMP}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Offline: {os.environ.get('TRANSFORMERS_OFFLINE')}")
    logger.info(f"Config: {json.dumps(cfg)}")

    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_BERT, local_files_only=True)
    tdir = os.path.join(GLUE_DIR, cfg["dir"])
    logger.info("Loading + pre-tokenizing data...")
    train_ds = GlueDataset(os.path.join(tdir, "train.tsv"), cfg, tokenizer, logger)
    dev_ds   = GlueDataset(os.path.join(tdir, "dev.tsv"),   cfg, tokenizer, logger)
    logger.info(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")
    if len(train_ds) == 0:
        logger.error("0 training rows — check column indices / has_header for this task")
        return

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                              persistent_workers=True)
    dev_loader   = DataLoader(dev_ds, batch_size=cfg["batch"]*2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    teacher, teacher_score, t_params = train_teacher(
        cfg, task, train_loader, dev_loader, device, tokenizer, logger)
    student, student_score, s_params = distill_student(
        cfg, task, teacher, t_params, train_loader, dev_loader, device, tokenizer, logger)

    logger.info("="*70)
    logger.info(f"SUMMARY ({task})  metric={cfg['metric']}")
    logger.info("="*70)
    logger.info(f"Teacher (12-layer) score : {teacher_score:.4f}  params {t_params:,}")
    logger.info(f"Student (6-block)  score : {student_score:.4f}  params {s_params:,} "
                f"({100*s_params/t_params:.1f}% of teacher)")
    if teacher_score > 0:
        logger.info(f"Score retained           : {100*student_score/teacher_score:.1f}%")
    logger.info(f"Best teacher -> {os.path.join(BEST_ROOT, task, 'teacher')}")
    logger.info(f"Best student -> {os.path.join(BEST_ROOT, task, 'student')}")


if __name__ == "__main__":
    main()

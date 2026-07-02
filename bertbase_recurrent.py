"""
Gated-recurrent weight-sharing architecture for BERT-base-uncased.

Same design as the DistilBERT / TinyBERT ports:
  - 12 BertLayers -> 6 GatedRecurrentBlocks, each run twice (12 effective passes)
  - one shared BertLayer per block, applied recurrently (pass_a -> pass_b)
  - a learned per-token gate blends pass_b into pass_a
  - gate bias init = +2.0 so the recurrent pass is used from the start
  - each block's shared layer seeded from a SINGLE teacher layer (no averaging)

BERT-family specifics (identical to the TinyBERT port):
  - layers live in  model.encoder.layer
  - each layer is a BertLayer with a tuple-returning forward
  - the attention mask is ALREADY extended (B,1,1,S) by BertModel and passed
    straight through to each layer -> do NOT re-extend inside the block

Target model: bert-base-uncased  (hidden=768, layers=12, heads=12, ffn=3072)
Fully local / offline.
"""

import os

# Hard offline backstop: refuse network access even if a flag is ever missed.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel,
    BertForSequenceClassification,
)

# Default local path for the downloaded BERT-base-uncased checkpoint.
LOCAL_MODEL = "./local_bertbaseuncased"

# 12 teacher layers -> 6 blocks. Seed each block from a single teacher layer,
# taking the deeper layer of each consecutive pair: (0,1)->1, (2,3)->3, ...
NUM_BLOCKS   = 6
SEED_SOURCES = [1, 3, 5, 7, 9, 11]


# ---------------------------------------------------------------------------
# Recurrent shared block with gate  (identical logic to prior ports)
# ---------------------------------------------------------------------------
class GatedRecurrentBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        from transformers.models.bert.modeling_bert import BertLayer
        self.shared_layer = BertLayer(config)
        self.gate = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # attention_mask is the EXTENDED mask (B,1,1,S) from BertModel.
        # BertLayer.forward returns a tuple; [0] is the hidden state.
        pass_a = self.shared_layer(hidden_states, attention_mask)[0]
        pass_b = self.shared_layer(pass_a, attention_mask)[0]
        gate_weights = torch.sigmoid(self.gate(pass_a))          # (B, S, 1)
        output = gate_weights * pass_b + (1.0 - gate_weights) * pass_a
        return (output,)


# ---------------------------------------------------------------------------
# Encoder replacement
# ---------------------------------------------------------------------------
def _install_recurrent_encoder(bert_model: BertModel, config: BertConfig):
    bert_model.encoder.layer = nn.ModuleList(
        [GatedRecurrentBlock(config) for _ in range(NUM_BLOCKS)]
    )


# ---------------------------------------------------------------------------
# Gate init: +2.0 bias so pass_b is used from the start.
# Must run AFTER post_init(), which re-initializes every Linear.
# ---------------------------------------------------------------------------
def _init_gates(blocks):
    for block in blocks:
        nn.init.constant_(block.gate.bias, 2.0)
        nn.init.normal_(block.gate.weight, std=0.02)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class CustomBertModel(BertModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        _install_recurrent_encoder(self, config)
        self.post_init()
        _init_gates(self.encoder.layer)


# ---------------------------------------------------------------------------
# Sequence-classification head (for GLUE fine-tuning)
# ---------------------------------------------------------------------------
class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        _install_recurrent_encoder(self.bert, config)
        self.post_init()
        _init_gates(self.bert.encoder.layer)


# ---------------------------------------------------------------------------
# Build a student seeded from a pretrained 12-layer BERT-base teacher.
#   block i  <- teacher layer SEED_SOURCES[i]   (single layers, no averaging)
# ---------------------------------------------------------------------------
def build_student_for_classification(local_path=LOCAL_MODEL, num_labels=2, problem_type=None):
    teacher = BertForSequenceClassification.from_pretrained(
        local_path, num_labels=num_labels, local_files_only=True
    )
    config = teacher.config
    config.num_labels = num_labels
    if problem_type:
        config.problem_type = problem_type     # "regression" for STS-B

    student = CustomBertForSequenceClassification(config)

    # Embeddings verbatim
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())

    # Seed shared layers from single teacher layers 1,3,5,7,9,11
    for s_idx, t_idx in enumerate(SEED_SOURCES):
        student.bert.encoder.layer[s_idx].shared_layer.load_state_dict(
            teacher.bert.encoder.layer[t_idx].state_dict()
        )

    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    del teacher
    return student, s_params, t_params


if __name__ == "__main__":
    # Smoke test with a fresh config (no weights needed)
    cfg = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, num_labels=2,
    )
    model = CustomBertForSequenceClassification(cfg)
    n_blocks = len(model.bert.encoder.layer)
    params = sum(p.numel() for p in model.parameters())
    print(f"blocks: {n_blocks} (each x2 = {2*n_blocks} effective passes)")
    print(f"params: {params:,}")

    ids = torch.randint(0, 30522, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    out = model(input_ids=ids, attention_mask=mask)
    print("logits shape:", out.logits.shape)   # (2, 2)

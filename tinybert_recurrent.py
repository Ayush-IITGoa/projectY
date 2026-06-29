"""
Gated-recurrent weight-sharing architecture ported to TinyBERT (4-layer).

Same design as the DistilBERT version:
  - 4 BertLayers  ->  2 GatedRecurrentBlocks, each run twice (4 effective passes)
  - one shared BertLayer per block, applied recurrently (pass_a -> pass_b)
  - a learned per-token gate blends pass_b into pass_a
  - gate bias init = +2.0 so the recurrent pass is used from the start
  - each block's shared layer seeded from a SINGLE teacher layer (no averaging)

Differences vs the DistilBERT port (BERT-family, not DistilBERT):
  - layers live in  model.encoder.layer  (not transformer.layer)
  - each layer is a BertLayer with a tuple-returning forward
  - the attention mask is ALREADY extended (B,1,1,S) by BertModel and passed
    straight through to each layer -> do NOT re-extend inside the block

Target model: huawei-noah/TinyBERT_General_4L_312D  (hidden=312, layers=4)
"""

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel,
    BertForSequenceClassification,
)


# ---------------------------------------------------------------------------
# Recurrent shared block with gate  (identical logic to the DistilBERT version)
# ---------------------------------------------------------------------------
class GatedRecurrentBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        # Import here so this file works whether the class lives in modeling_bert
        # or a vendored copy; BertLayer is the standard encoder layer.
        from transformers.models.bert.modeling_bert import BertLayer
        self.shared_layer = BertLayer(config)
        # Gate: hidden -> 1, sigmoid -> per-token mixing weight in (0,1)
        self.gate = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # attention_mask here is the EXTENDED mask (B,1,1,S) from BertModel.
        # BertLayer.forward returns a tuple; [0] is the hidden state.

        # Pass A
        pass_a = self.shared_layer(hidden_states, attention_mask)[0]

        # Pass B: same weights, applied to pass_a (the recurrent step)
        pass_b = self.shared_layer(pass_a, attention_mask)[0]

        # Gate conditioned on pass_a, blend pass_b into pass_a
        gate_weights = torch.sigmoid(self.gate(pass_a))          # (B, S, 1)
        output = gate_weights * pass_b + (1.0 - gate_weights) * pass_a

        # Return a tuple to stay compatible with BertEncoder's expectations
        return (output,)


# ---------------------------------------------------------------------------
# Encoder replacement: swap the 4-layer stack for 2 gated-recurrent blocks
# ---------------------------------------------------------------------------
def _install_recurrent_encoder(bert_model: BertModel, config: BertConfig):
    bert_model.encoder.layer = nn.ModuleList(
        [GatedRecurrentBlock(config) for _ in range(2)]
    )


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class CustomTinyBertModel(BertModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        _install_recurrent_encoder(self, config)
        self.post_init()
        _init_gates(self.encoder.layer)


# ---------------------------------------------------------------------------
# Sequence-classification head (for GLUE fine-tuning)
# ---------------------------------------------------------------------------
class CustomTinyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        _install_recurrent_encoder(self.bert, config)
        self.post_init()
        _init_gates(self.bert.encoder.layer)


# ---------------------------------------------------------------------------
# Gate init: +2.0 bias so pass_b is used from the start.
# Must run AFTER post_init(), which re-initializes every Linear.
# ---------------------------------------------------------------------------
def _init_gates(blocks):
    for block in blocks:
        nn.init.constant_(block.gate.bias, 2.0)
        nn.init.normal_(block.gate.weight, std=0.02)


# ---------------------------------------------------------------------------
# Build a student seeded from a pretrained 4-layer TinyBERT teacher.
#   block 0  <- teacher layer 1
#   block 1  <- teacher layer 3
# (single layers, the deeper of each pair; no averaging)
# ---------------------------------------------------------------------------
def build_student_for_classification(local_path, num_labels, problem_type=None):
    teacher = BertForSequenceClassification.from_pretrained(
        local_path, num_labels=num_labels, local_files_only=True
    )
    config = teacher.config
    config.num_labels = num_labels
    if problem_type:
        config.problem_type = problem_type     # "regression" for STS-B

    student = CustomTinyBertForSequenceClassification(config)

    # Embeddings verbatim
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())

    # Seed shared layers from teacher layers 1 and 3
    src = [1, 3]
    for s_idx, t_idx in enumerate(src):
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
        vocab_size=30522, hidden_size=312, num_hidden_layers=4,
        num_attention_heads=12, intermediate_size=1200, num_labels=2,
    )
    model = CustomTinyBertForSequenceClassification(cfg)
    n_blocks = len(model.bert.encoder.layer)
    params = sum(p.numel() for p in model.parameters())
    print(f"blocks: {n_blocks} (each x2 = {2*n_blocks} effective passes)")
    print(f"params: {params:,}")

    ids = torch.randint(0, 30522, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    out = model(input_ids=ids, attention_mask=mask)
    print("logits shape:", out.logits.shape)   # (2, 2)

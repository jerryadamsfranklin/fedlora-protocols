# Phase 2 Part 1 — 8B B-only fraction pre-check

**Status: PASS** (2026-07-16)

## Checks

| Check | Result |
|-------|--------|
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` only — **no MLP** |
| LoRA rank | `r=16` (matches TinyLlama / 3.2-3B) |
| Geometry | hidden=4096, heads=32/8 GQA, layers=32, head_dim=128 |
| B-only fraction | **38.46%** |
| Cross-scale spread | 36.00% / 40.00% / 38.46% → max−min **4.00 pp** (≤5 pp PASS) |

## Config lock-in

Before this check there was no 8B YAML. Created:

- `config/base_config_llama3_8b.yaml` — inherits `base_config_4layers.yaml`, re-declares the same four attention targets and `r=16`, model `meta-llama/Meta-Llama-3.1-8B`.

Experiment configs (FLoRA / RA / Two-Phase) should `_inherit: base_config_llama3_8b.yaml` the same way 3B experiments inherit `base_config_llama3_3b.yaml`.

## Arithmetic (reproduced via `scripts/compute_b_only_fraction.py`)

```
LLaMA-3.1-8B  LoRA A: 8,388,608   LoRA B: 5,242,880   B/(A+B)=38.46%
per-module: q=0.500, k=0.200, v=0.200, o=0.500
```

Rank cancels in the fraction; consistency of **target modules** across scales is what matters for the paper’s communication comparison.

## Verdict

Part 1 cleared. Safe to proceed to Part 2 environment / Part 3 matrix once HF access (confirmed by maintainer) and the Part 3.3 (a)/(b) decision are in place.

# Phase 2 Part 3.3(b) — TinyLlama α=0.1 baseline matrix

**Decision:** option **(b)** (maintainer 2026-07-16).

**Scope (strict):** 6 TinyLlama-1.1B runs only — FLoRA and Two-Phase K=8 × Non-IID
Dirichlet **α=0.1** × seeds **42 / 123 / 456**. No other scales, no α=0.5, no 8B.

**Does not block** Phase 2 8B setup; run in parallel on an RTX 4090.

**Critical:** full downstream eval on every run — MMLU, ARC-Easy, BoolQ, HellaSwag
(500 examples each, seed 42 for eval sampling). Comparative robustness claims need
baseline **downstream quality + final communication MB**, not training loss alone.

## Matrix

See `docs/phase2_part33b_manifest.csv`.

| # | Method | Config | Seeds |
|---|--------|--------|-------|
| 1–3 | FLoRA | `config/exp_flora_noniid_alpha01.yaml` | 42, 123, 456 |
| 4–6 | Two-Phase K=8 | `config/exp_two_phase_k8_noniid_alpha01.yaml` | 42, 123, 456 |

Tag: `phase1b_alpha01_baselines` (extends Phase 1 TinyLlama tier in the master CSV).

Compare against existing CUDA RA α=0.1 Phase 1 runs under
`exp_reverse_adaptive_noniid_alpha01/.../phase1_cuda_rerun/`.

## RTX 4090 — full setup + run

```bash
cd /workspace
git clone -b neurocomputing/phase-1-cuda-reruns \
  https://github.com/jerryadamsfranklin/fedlora-protocols.git
cd fedlora-protocols
# if Part 3.3(b) configs are not on that branch yet, pull / cherry-pick, or copy the two YAMLs

source /venv/main/bin/activate
pip install -r requirements.txt
python3 -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
# expect: torch 2.2.x, cuda True, transformers 4.45.2

tmux new -s part33b
bash scripts/run_part33b_alpha01_baselines.sh all
# Ctrl-b d to detach
```

Or step-wise:

```bash
bash scripts/run_part33b_alpha01_baselines.sh train
bash scripts/run_part33b_alpha01_baselines.sh eval
bash scripts/run_part33b_alpha01_baselines.sh summarize
python3 scripts/build_results_table.py
```

Manual one-liner pattern:

```bash
python3 scripts/run_experiment.py \
  --config config/exp_flora_noniid_alpha01.yaml \
  --seed 42 --device cuda --tag phase1b_alpha01_baselines --save-every 10

python3 scripts/evaluate_checkpoint.py \
  --checkpoint results/raw/exp_flora_noniid_alpha01/flora/seed_42/phase1b_alpha01_baselines/<TS>/final_adapter_state.pt \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --benchmarks mmlu arc_easy boolq hellaswag \
  --num-examples 500 --seed 42 --device cuda \
  --output results/downstream/exp_flora_noniid_alpha01/flora/seed_42/phase1b_alpha01_baselines/downstream_results.json
```

## Deliverable before non-IID section draft

Report for all 6 runs:

| Field | Source |
|-------|--------|
| final_communication_mb | `results.json` last round |
| final_avg_loss | `results.json` last round |
| switch_round (Two-Phase) | `aggregator_stats.switch_round` (expect ~9 / K=8) |
| mmlu / arc_easy / boolq / hellaswag accuracy | `downstream_results.json` |

Then rebuild `analysis/final_results_table.csv` via `scripts/build_results_table.py`
and paste the 6-row summary into this doc (status table below).

## Status

| # | Method | Seed | final_comm_mb | final_loss | switch | mmlu | arc_easy | boolq | hellaswag | Status |
|---|--------|------|---------------|------------|--------|------|----------|-------|-----------|--------|
| 1 | flora | 42 | | | — | | | | | pending |
| 2 | flora | 123 | | | — | | | | | pending |
| 3 | flora | 456 | | | — | | | | | pending |
| 4 | two_phase | 42 | | | | | | | | pending |
| 5 | two_phase | 123 | | | | | | | | pending |
| 6 | two_phase | 456 | | | | | | | | pending |

## Notes

- `scripts/evaluate_checkpoint.py` now accepts `--device cuda` (required on Vast).
- Downstream paths must mirror raw tag so `build_results_table.py` joins correctly.
- Do **not** expand to 3B/8B α=0.1 baselines without a new maintainer decision.

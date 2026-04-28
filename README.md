# Federated LoRA experiments

Comparison of federated fine-tuning approaches that aggregate client-side LoRA adapters. Training is simulated with a central orchestrator (single-process prototype): clients hold partitioned data; each round updates LoRA weights locally; the server applies the chosen aggregator.

Implemented aggregators:
- **Baselines**: **FedIT**, **FFA-LoRA**, **FLoRA**, **FlexLoRA**
- **Adaptive**: **FedLoRA-Adaptive v1** (`fedlora_adaptive`), **FedLoRA-Adaptive v2** (`fedlora_adaptive_v2`)
- **Novel methods (week plan)**: **Two-Phase** (`two_phase`), **Reverse-Adaptive** (`reverse_adaptive`), **Budget-Adaptive** (`budget_adaptive`), **Curriculum-Rank** (`curriculum_rank`)

Full methodology, research questions (RQ1–RQ5), experiment matrix, and publication checklist are in **`Federated_LoRA_Complete_Guide.md`** at the repo root.

## Research questions (summary)

| RQ | Topic |
|----|--------|
| **RQ1** | IID baseline — how do methods compare when data is evenly split? |
| **RQ2** | Non-IID — label skew vs quantity skew partitions |
| **RQ3** | Communication cost vs final loss |
| **RQ4** | LoRA rank sensitivity |
| **RQ5** | Scaling with number of clients |

Publication-style figures **`figures/fig1_convergence_comparison.pdf`** through **`fig5_client_scaling.pdf`** are generated from saved runs; see **Figures** below.

## Requirements

- **Python 3.10+** recommended  
- **PyTorch** — default device is **Apple MPS** when available, otherwise **CPU** (`scripts/run_experiment.py`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: add a **`.env`** with Hugging Face or Weights & Biases tokens if you use gated models or logging.

Local handoff bundles for external tools (e.g. Claude) belong under **`exports/`** or **`docs/HANDOFF_*.md`**; those paths are listed in **`.gitignore`** and will not be committed.

## Quick start — one run

From the repository root:

```bash
python scripts/run_experiment.py --config config/revised_exp1_iid.yaml --method fedit
```

Aggregation methods (most common): `fedit`, `ffa_lora`, `flora`, `flexlora`, `fedlora_adaptive`, `fedlora_adaptive_v2`, `two_phase`, `reverse_adaptive`, `budget_adaptive`, `curriculum_rank`.

Sweeps (used in EXP4 / EXP5):

```bash
python scripts/run_experiment.py --config config/revised_exp4_rank.yaml --method fedit --lora_r 16
python scripts/run_experiment.py --config config/revised_exp5_scaling.yaml --method flora --num_clients 10
```

Configs use YAML `_inherit` merging (e.g. `revised_exp1_iid.yaml` → `revised_base.yaml` → `base_config.yaml`).

## Full revised protocol (all experiments)

To run the full **revised** suite sequentially (EXP1–EXP5 as defined in `run_all_experiments.sh`):

```bash
bash scripts/run_all_experiments.sh
```

Optional: set `CONDA_ENV` (default `fedlora`) and `LOG_DIR` (default `logs`). Logs are written under `logs/` as one file per job.

Runtime is long; runs are intended for overnight or cluster-style execution.

## Results layout

Each run writes a timestamped directory under **`results/raw/`**:

```text
results/raw/<experiment>/<method>/seed_<seed>[_r<rank>][_c<clients>]/[run_NN_of_MM/]<YYYYMMDD_HHMMSS>/
  results.json
  config_merged.yaml
  run_meta.json
```

`results.json` is a list of per-round records (`round`, `avg_loss`, `communication_mb`, `round_time`, …) used by analysis and plotting.

The optional `run_NN_of_MM/` folder is used by batch scripts to make it easy to locate “run 7/12” etc.

## Novel methods (Two-Phase / Reverse-Adaptive / Budget-Adaptive / Curriculum-Rank)

The novel-method protocol and rationale are described in `NOVEL_METHODS_IMPLEMENTATION.md` (local notes). The key config for meaningful per-layer / 4-projection LoRA experiments is:
- `config/base_config_4layers.yaml` (LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`)

### Run the 12-run novel-method batch (seed 42)

```bash
SEED=42 ./scripts/run_novel_methods.sh
```

This runs:
- Two-Phase sweep (`exp_two_phase_k5`, `k8`, `k10`, `k12`)
- Reverse-Adaptive (`exp_reverse_adaptive`)
- Budget-Adaptive sweep (`exp_budget_800`, `1200`, `1600`, `2000`)
- Baselines on the same 4-layer setup (`flora`, `ffa_lora`, `fedit`)

### Run curriculum-rank schedule sweep (seed 42)

```bash
python3 scripts/run_experiment.py --config config/exp_curriculum_default.yaml --method curriculum_rank --seed 42
python3 scripts/run_experiment.py --config config/exp_curriculum_2stage.yaml  --method curriculum_rank --seed 42
python3 scripts/run_experiment.py --config config/exp_curriculum_r8start.yaml --method curriculum_rank --seed 42
python3 scripts/run_experiment.py --config config/exp_curriculum_gradual.yaml --method curriculum_rank --seed 42
```

Curriculum-rank uses **client-side** gradient masking at the scheduled rank and **sliced LoRA uploads**; the server aggregates and pads back to full rank for compatibility.

### Final experiments (Two-Phase paper — multi-seed + non-IID)

18 runs (3 seeds × 6 configs): IID Two-Phase K=8/K=10, FLoRA IID baseline, then the same three methods with **label_skew** Dirichlet **partition_alpha=0.5** (Alpaca uses instruction-length proxy labels when no `label` column exists).

```bash
./scripts/run_final_experiments.sh
```

Summarize all runs under `results/raw/`:

```bash
python3 scripts/analyze_final_results.py
```

## Analysis and figures

After you have **`results/*/results.json`** files:

```bash
python scripts/analyze_results.py
python scripts/generate_figures.py
```

- **`analysis/`** — aggregated tables (`all_results*.csv`, `comparison_table.csv`, `statistical_tests.csv`) and **`RESEARCH_SUMMARY.md`** (narrative aligned with RQ1–RQ5). When both legacy and `revised_*` runs exist, summaries prefer **revised** protocol rows.
- **`figures/`** — `fig1`–`fig5` (PDF + PNG), plus legacy aliases `convergence_exp1`–`exp3`.

Regenerate figures whenever you add new result directories or change plotting code.

## Repository layout (high level)

| Path | Purpose |
|------|---------|
| `config/` | YAML configs (`revised_*` = paper protocol; `exp*` without prefix = older / exploratory) |
| `src/` | Model (`lora_model`), data partitioning, federation client/server, aggregators |
| `scripts/` | `run_experiment.py`, `run_all_experiments.sh`, `analyze_results.py`, `generate_figures.py` |
| `results/` | Run outputs (git may omit large runs; regenerate locally) |
| `analysis/` | Generated tables and research summary |
| `figures/` | Generated figures |

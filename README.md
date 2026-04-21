# Federated LoRA experiments

Comparison of four federated fine-tuning approaches that aggregate client-side LoRA adapters: **FedIT**, **FFA-LoRA**, **FLoRA**, and **FlexLoRA**. Training is simulated with a central orchestrator (single-process prototype): clients hold partitioned data; each round updates LoRA weights locally; the server applies the chosen aggregator.

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

Optional: copy **`.env.example`** to **`.env`** for Hugging Face or Weights & Biases tokens if you use gated models or logging (see comments in `.env.example`).

## Quick start — one run

From the repository root:

```bash
python scripts/run_experiment.py --config config/revised_exp1_iid.yaml --method fedit
```

Aggregation methods: `fedit`, `ffa_lora`, `flora`, `flexlora`.

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

Each run writes a timestamped directory under **`results/`**:

```text
results/<experiment_name>_<method>[_r<rank>][_c<clients>]_<YYYYMMDD_HHMMSS>/results.json
```

`results.json` is a list of per-round records (`round`, `avg_loss`, `communication_mb`, `round_time`, …) used by analysis and plotting.

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

## License and citation

Add your license and citation text when you publish the paper or dataset.

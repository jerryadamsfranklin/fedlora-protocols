# fedlora-protocols

Communication-efficient federated fine-tuning for large language models, with measured (not theoretical) communication accounting and an adaptive phase-switching aggregator.

## Overview

This repository accompanies the paper *Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning*. It implements six federated LoRA aggregation methods, evaluates them on Alpaca-3k and Dolly-15k instruction-tuning at two model scales (TinyLlama-1.1B and LLaMA-3.2-3B), and reports per-round byte-tracked communication costs measured at the transport layer.

### Headline results

- Five published protocols are placed on one measured communication-quality frontier at four distinct operating points.
- **ReverseAdaptive** achieves 40.5 percent measured round-trip communication savings versus FLoRA on TinyLlama-1.1B, at a held-out instruction-following loss cost of 0.0063.
- The frontier has a knee: continuing past that point to FFA-LoRA buys 21.3 further percentage points of savings at roughly five times the quality cost per point.
- **Two-Phase K=8** achieves 27.7 percent measured savings on TinyLlama-1.1B.
- Federated methods cluster within 0.5 percentage points on LLaMA-3.2-3B downstream accuracy and within 1.0 percentage point on TinyLlama-1.1B, where the zero-shot benchmarks do not discriminate between protocols.
- A no-switch sanity baseline reproduces the matched FLoRA run to 10 decimal places within a single backend, confirming the adaptive wrapper introduces no perturbation when its switching logic is disabled.

## Aggregators implemented

| Aggregator | Description | Reference |
|------------|-------------|-----------|
| FedIT | LoRA aggregation via FedAvg over A and B | Zhang et al., 2024 |
| FFA-LoRA | Frozen A, aggregate only B | Sun et al., 2024 |
| FLoRA | Stack-and-SVD aggregation of BA products | Wang et al., 2024 |
| FlexLoRA | Heterogeneous-rank LoRA aggregation | Bai et al., 2024 |
| Two-Phase | FLoRA for K rounds, then FFA-LoRA | this paper |
| ReverseAdaptive | Loss-plateau-triggered switch from FLoRA to FFA-LoRA | this paper |

## Repository layout

```
fedlora-protocols/
├── src/
│   ├── federation/         # Server, client, aggregators
│   ├── models/             # LoRA model wrapper (PEFT-based)
│   ├── data/               # Alpaca loader, partitioners (IID, Dirichlet)
│   └── evaluation/         # Benchmarks, scorers, metrics
├── config/                 # YAML experiment configs (with _inherit)
├── scripts/                # Runners, analysis, figure generation
├── tests/                  # 20 unit tests covering aggregators and runner
├── results/                # Per-run JSONs (raw/) and downstream eval (downstream/)
├── analysis/               # Master CSVs and statistical tests
├── figures/                # Paper-ready PDFs
└── docs/                   # Method notes and paper outline
```

## Quick start

```bash
git clone https://github.com/jerryadamsfranklin/fedlora-protocols.git
cd fedlora-protocols
pip install -r requirements.txt

# Run a single experiment
python3 scripts/run_experiment.py \
    --config config/exp_two_phase_k8.yaml \
    --seed 42 \
    --tag my_run

# Evaluate a saved checkpoint on downstream benchmarks
python3 scripts/evaluate_checkpoint.py \
    --checkpoint results/raw/.../final_adapter_state.pt \
    --num-examples 500 \
    --output results/downstream/my_eval.json
```

## Reproducing paper figures

```bash
python3 scripts/build_results_table.py
python3 scripts/generate_paper_figures.py
ls figures/  # six PDFs corresponding to paper figures
```

## Hardware

The primary experiment corpus was produced on Apple M4 Pro (48 GB unified memory) with the MPS backend and reproduced on rented NVIDIA hardware (RTX 4090 for TinyLlama-1.1B, A100 40 GB for LLaMA-3.2-3B). LLaMA-3.2-3B uses float16 base weights with float32 LoRA parameters for numerical stability; this is configured automatically by `config/base_config_llama3_3b.yaml`.

## Citation

If you use this codebase, please cite the paper:

```
@article{franklin2026fedlora,
  title  = {Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning},
  author = {Franklin, Jerry Adams},
  year   = {2026},
  journal = {Neurocomputing},
  note   = {Manuscript under review}
}
```

(Update with the actual citation once the paper is published.)

## License

MIT (see LICENSE file). Reuse for research and production is welcome; please cite the paper if the work informs published results.

## Acknowledgments

The author used Claude (Anthropic) as a development and writing assistant. All technical content, methodology, and results are the author's own.

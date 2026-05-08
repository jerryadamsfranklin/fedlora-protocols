# fedlora-protocols

Communication-efficient federated fine-tuning for large language models, with measured (not theoretical) communication accounting and an adaptive phase-switching aggregator.

## Overview

This repository accompanies the paper *Adaptive Phase-Switching for Communication-Efficient Federated LoRA*. It implements four federated LoRA aggregation methods, evaluates them on Alpaca-3k instruction-tuning at two model scales (TinyLlama-1.1B and LLaMA-3.2-3B), and reports per-round byte-tracked communication costs and downstream zero-shot benchmark accuracy.

### Headline results

- **Two-Phase K=8** achieves 27.7 percent measured round-trip communication savings versus full-rank federated LoRA on TinyLlama-1.1B.
- **ReverseAdaptive** achieves 40.5 percent measured savings via single-parameter threshold tuning, recovering the entire fixed-K Pareto frontier.
- All federated methods cluster within 1.0 percentage point on LLaMA-3.2-3B downstream accuracy across MMLU, ARC-Easy, BoolQ, and HellaSwag.
- A no-switch sanity baseline produces bit-identical results to plain FLoRA, confirming the adaptive wrapper introduces zero observable perturbation when its switching logic is disabled.

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
python3 scripts/statistical_analysis.py
python3 scripts/generate_paper_figures.py
ls figures/  # six PDFs corresponding to paper figures
```

## Hardware

The codebase was developed and tested on Apple M4 Pro (48 GB unified memory) with the MPS backend. CUDA is supported via PyTorch's standard device selection but has not been validated by the maintainer. LLaMA-3.2-3B requires float16 base weights with float32 LoRA parameters on MPS for numerical stability; this is configured automatically by `config/base_config_llama3_3b.yaml`.

## Citation

If you use this codebase, please cite the paper:

```
@article{franklin2026fedlora,
  title  = {Adaptive Phase-Switching for Communication-Efficient Federated LoRA},
  author = {Franklin, Jerry Adams},
  year   = {2026},
  note   = {Manuscript in preparation, Transactions on Machine Learning Research}
}
```

(Update with the actual citation once the paper is published.)

## License

MIT (see LICENSE file). Reuse for research and production is welcome; please cite the paper if the work informs published results.

## Acknowledgments

The author used Claude (Anthropic) as a development and writing assistant. All technical content, methodology, and results are the author's own.

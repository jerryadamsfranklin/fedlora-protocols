# Analysis data

Which file backs which table and figure in the paper.

| File | Backs |
|---|---|
| `final_results_table.csv` | Table 7 (LLaMA-3.2-3B), Table S3; input to `generate_paper_figures.py` |
| `qv_only_cuda_per_seed_runs.csv` | Table 2, Table 3, Table S1, Table S4 (loss and communication) |
| `neuro_part12_qvonly_holdout_strict24.csv` | Held-out deltas for FedIT, FFA-LoRA, and all Dolly-15k rows |
| `neuro_existing_alpaca_holdout_phase1_only.csv` | Held-out deltas for FLoRA, Two-Phase, and ReverseAdaptive on Alpaca |

Regenerate figures and tables from these CSVs:

```
python3 scripts/build_results_table.py
python3 scripts/generate_paper_figures.py
python3 scripts/verify_numbers.py --analysis-dir analysis
```

## LoRA target modules

All results reported in the paper use `q_proj` and `v_proj`, as stated in
Table 1 of the manuscript and in `config/base_config.yaml`.

Some run records under `results/raw/` and some rows of
`qv_only_cuda_per_seed_runs.csv` carry a `target_modules` field of
`q_proj,k_proj,v_proj,o_proj`. That field records a superseded configuration
generation and does not describe the runs the paper reports. The measured byte
counts are the check. A q_proj and v_proj LoRA state on TinyLlama-1.1B is
2,252,800 parameters over 22 layers, or 9,011,200 bytes per client per
direction. The released run records show exactly 85.94 MB per direction per
round across 10 clients, which is that state, and a cumulative total of
2578.12 MB over 15 rounds, which is the value reported in Table 2. A
four-projection target set would double both figures. The runs therefore used
q_proj and v_proj, whatever the `target_modules` string records.

Note that `communication_mb`, `upload_mb`, and `download_mb` in
`results/raw/**/results.json` are cumulative across rounds, not per-round. The
final round's value is the run total.

The 36 percent B-only fraction derived in Section III-D likewise holds only for
`q_proj` and `v_proj`.

## Superseded files

`legacy/` holds earlier analysis outputs retained for provenance. They do not
back any number in the paper and in places disagree with it. In particular
`legacy/neuro_part1_dolly_holdout.csv` and
`legacy/neuro_part2_baselines_holdout.csv` were produced under the superseded
four-projection target set and report held-out values that differ from Table 3.
Use `neuro_part12_qvonly_holdout_strict24.csv` instead.

## Run provenance

`results/MANIFEST.csv` maps every run that backs a reported number to the table,
figure, or section it supports, and records the git commit each run was produced
at. `results/README.md` describes the aggregation hazards that manifest
membership guards against. Runs in `results/raw/` that are not listed in the
manifest are development history and do not back any reported result.

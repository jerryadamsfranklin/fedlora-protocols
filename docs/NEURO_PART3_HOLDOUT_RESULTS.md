# Neurocomputing Part 3 Holdout Results (Part 1 + Part 2)

This note summarizes held-out instruction-following evaluation for the 24 newly completed runs:

- Part 1 (Dolly): 12 runs
- Part 2 (published baselines on Alpaca): 12 runs

Evaluation source files:

- `analysis/neuro_part1_dolly_holdout.csv`
- `analysis/neuro_part2_baselines_holdout.csv`

## Metric definition

- `delta_loss = tuned_loss - base_loss` (lower is better; negative means improvement over base model).
- `delta_ppl = tuned_perplexity - base_perplexity` (lower is better; negative means improvement).
- Reported values are mean +/- population std over 3 seeds.

## Aggregated Results

| Experiment | n | Comm MB | Tuned Loss | Base Loss | Delta Loss | Tuned PPL | Base PPL | Delta PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp_dolly_flora_iid` | 3 | 5156.25 | 1.7169 +/- 0.0006 | 2.2608 | -0.5439 +/- 0.0006 | 5.5674 | 9.5905 | -4.0232 |
| `exp_dolly_two_phase_k8_iid` | 3 | 3726.25 | 1.7183 +/- 0.0005 | 2.2608 | -0.5424 +/- 0.0005 | 5.5752 | 9.5905 | -4.0153 |
| `exp_dolly_reverse_adaptive_iid` | 3 | 3066.25 | 1.7195 +/- 0.0005 | 2.2608 | -0.5413 +/- 0.0005 | 5.5815 | 9.5905 | -4.0090 |
| `exp_dolly_reverse_adaptive_noniid_alpha05` | 3 | 3066.25 | 1.7206 +/- 0.0008 | 2.2608 | -0.5402 +/- 0.0008 | 5.5877 | 9.5905 | -4.0029 |
| `exp_fedit_iid` | 3 | 5156.25 | 1.3260 +/- 0.0005 | 1.9352 | -0.6092 +/- 0.0005 | 3.7660 | 6.9254 | -3.1595 |
| `exp_fedit_noniid_alpha05` | 3 | 5156.25 | 1.3355 +/- 0.0021 | 1.9352 | -0.5996 +/- 0.0021 | 3.8021 | 6.9254 | -3.1233 |
| `exp_ffa_lora_iid` | 3 | 1966.25 | 1.3391 +/- 0.0002 | 1.9352 | -0.5961 +/- 0.0002 | 3.8158 | 6.9254 | -3.1097 |
| `exp_ffa_lora_noniid_alpha05` | 3 | 1966.25 | 1.3340 +/- 0.0009 | 1.9352 | -0.6012 +/- 0.0009 | 3.7964 | 6.9254 | -3.1291 |

## Interpretation (for paper text)

- All 24 runs improve over base model on held-out instruction-following (`delta_loss < 0`, `delta_ppl < 0`).
- Improvements are highly stable across seeds (small std in all groups).
- Cross-dataset absolute losses should be compared within dataset blocks (Dolly vs Alpaca) since base losses differ by dataset.

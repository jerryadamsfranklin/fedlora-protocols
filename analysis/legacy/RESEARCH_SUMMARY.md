# Federated LoRA experiments — summary (revised protocol)
This note ties numeric results to the research questions in `Federated_LoRA_Complete_Guide.md`. Rows use single seeds; treat **statistical_tests.csv** as exploratory unless you add multi-seed runs.

## RQ1 — IID baseline (revised EXP1)
| Method | Final loss | Comm (MB) |
|---|---:|---:|
| fedit | 1.2316 | 1289.06 |
| ffa_lora | 1.2796 | 464.06 |
| flexlora | 1.2189 | 1289.06 |
| flora | 1.2189 | 1289.06 |

Lowest final loss on IID: **flexlora** (1.2189). FFA-LoRA shows ~50% lower communication than full LoRA methods at comparable rank.

## RQ2 — Non-IID (revised EXP2 label skew, EXP3 quantity skew)
### Label skew (CommonsenseQA)
| Method | Final loss | Comm (MB) |
|---|---:|---:|
| fedit | 2.7501 | 1289.06 |
| ffa_lora | 2.9317 | 464.06 |
| flexlora | 2.7095 | 1289.06 |
| flora | 2.7095 | 1289.06 |

### Quantity skew (Alpaca)
| Method | Final loss | Comm (MB) |
|---|---:|---:|
| fedit | 1.2516 | 1289.06 |
| ffa_lora | 1.2891 | 464.06 |
| flexlora | 1.2322 | 1289.06 |
| flora | 1.2322 | 1289.06 |


## RQ3 — Communication cost
See IID table above: **fedit / flora / flexlora** track **~1289 MB** cumulative (full LoRA exchange); **ffa_lora** ~**464 MB** (B matrices only). FLoRA and FlexLoRA match under the current weighted $\Delta W$ + SVD implementation.

## RQ4 — Rank sensitivity (revised EXP4)
| Method | Rank | Final loss | Comm (MB) |
|---|---:|---:|---:|
| fedit | 8.0 | 1.2297 | 644.53 |
| fedit | 16.0 | 1.2263 | 1289.06 |
| fedit | 32.0 | 1.2282 | 2578.12 |
| ffa_lora | 8.0 | 1.2800 | 232.03 |
| ffa_lora | 16.0 | 1.2777 | 464.06 |
| ffa_lora | 32.0 | 1.2783 | 928.12 |

Communication scales ~linearly with rank; final loss changes are small across $r \in \{8,16,32\}$ in this pilot, suggesting **diminishing returns past $r=16$** here.

## RQ5 — Client scaling (revised EXP5)
| Method | Clients | Final loss | Comm (MB) |
|---|---:|---:|---:|
| fedit | 5.0 | 1.1761 | 644.53 |
| fedit | 10.0 | 1.2316 | 1289.06 |
| flora | 5.0 | 1.1450 | 644.53 |
| flora | 10.0 | 1.2189 | 1289.06 |

With fixed total samples, **fewer clients → more data per client**; final loss typically improves from 10 → 5 clients. Total communication roughly halves when halving participants per round.

## Figures

Run `python scripts/generate_figures.py` to regenerate `figures/fig1_convergence_comparison.pdf` through `fig5_client_scaling.pdf`.

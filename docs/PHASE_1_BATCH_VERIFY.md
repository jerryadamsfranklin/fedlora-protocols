# Phase 1 batch MPS↔CUDA verification

**SUMMARY: PASS=27  FAIL=7  MISSING=0  TOTAL=34**

Tolerance: IID `atol=0.01`; Non-IID `atol=0.025` + exact switch + ≤2 consecutive `|Δ|>0.01`.

| Order | Status | Exp | Setting | Seed | Fail checks |
|------:|:------:|-----|---------|-----:|------------:|
| 1 | **PASS** | `exp_reverse_adaptive_noniid_alpha05` | noniid_alpha0.5 | 42 | 0 |
| 2 | **PASS** | `exp_flora_iid` | iid | 42 | 0 |
| 3 | **PASS** | `exp_flora_iid` | iid | 123 | 0 |
| 4 | **PASS** | `exp_flora_iid` | iid | 456 | 0 |
| 5 | **PASS** | `exp_flora_noniid_alpha05` | noniid_alpha0.5 | 42 | 0 |
| 6 | **PASS** | `exp_flora_noniid_alpha05` | noniid_alpha0.5 | 123 | 0 |
| 7 | **PASS** | `exp_flora_noniid_alpha05` | noniid_alpha0.5 | 456 | 0 |
| 8 | **PASS** | `exp_reverse_adaptive_iid` | iid | 42 | 0 |
| 9 | **PASS** | `exp_reverse_adaptive_iid` | iid | 123 | 0 |
| 10 | **PASS** | `exp_reverse_adaptive_iid` | iid | 456 | 0 |
| 11 | **FAIL** | `exp_reverse_adaptive_threshold_ablation` | iid_threshold_tau=0.001 | 42 | 1 |
| 12 | **FAIL** | `exp_reverse_adaptive_threshold_ablation` | iid_threshold_tau=0.002 | 42 | 1 |
| 13 | **PASS** | `exp_reverse_adaptive_threshold_ablation` | iid_threshold_tau=0.005 | 42 | 0 |
| 14 | **PASS** | `exp_reverse_adaptive_threshold_ablation` | iid_threshold_tau=0.01 | 42 | 0 |
| 15 | **PASS** | `exp_reverse_adaptive_noniid_alpha01` | noniid_alpha0.1 | 42 | 0 |
| 16 | **FAIL** | `exp_reverse_adaptive_noniid_alpha01` | noniid_alpha0.1 | 123 | 1 |
| 17 | **FAIL** | `exp_reverse_adaptive_noniid_alpha01` | noniid_alpha0.1 | 456 | 6 |
| 18 | **PASS** | `exp_reverse_adaptive_noniid_alpha05` | noniid_alpha0.5 | 123 | 0 |
| 19 | **PASS** | `exp_reverse_adaptive_noniid_alpha05` | noniid_alpha0.5 | 456 | 0 |
| 20 | **PASS** | `exp_two_phase_k8` | iid | 42 | 0 |
| 21 | **PASS** | `exp_two_phase_k8` | iid | 123 | 0 |
| 22 | **PASS** | `exp_two_phase_k8` | iid | 456 | 0 |
| 23 | **FAIL** | `exp_two_phase_k8_noniid_alpha05` | noniid_alpha0.5 | 42 | 2 |
| 24 | **FAIL** | `exp_two_phase_k8_noniid_alpha05` | noniid_alpha0.5 | 123 | 2 |
| 25 | **FAIL** | `exp_two_phase_k8_noniid_alpha05` | noniid_alpha0.5 | 456 | 2 |
| 26 | **PASS** | `exp_llama3_flora_iid` | iid | 42 | 0 |
| 27 | **PASS** | `exp_llama3_flora_iid` | iid | 123 | 0 |
| 28 | **PASS** | `exp_llama3_flora_iid` | iid | 456 | 0 |
| 29 | **PASS** | `exp_llama3_reverse_adaptive_iid` | iid | 42 | 0 |
| 30 | **PASS** | `exp_llama3_reverse_adaptive_iid` | iid | 123 | 0 |
| 31 | **PASS** | `exp_llama3_reverse_adaptive_iid` | iid | 456 | 0 |
| 32 | **PASS** | `exp_llama3_two_phase_k8_iid` | iid | 42 | 0 |
| 33 | **PASS** | `exp_llama3_two_phase_k8_iid` | iid | 123 | 0 |
| 34 | **PASS** | `exp_llama3_two_phase_k8_iid` | iid | 456 | 0 |

## Failures

- **#11** `exp_reverse_adaptive_threshold_ablation` seed=42: [FAIL] final_communication_mb: mps=2083.125000  cuda=1973.125000  |Δ|=110.000000  (atol=0.01)
- **#12** `exp_reverse_adaptive_threshold_ablation` seed=42: [FAIL] final_communication_mb: mps=1863.125000  cuda=1753.125000  |Δ|=110.000000  (atol=0.01)
- **#16** `exp_reverse_adaptive_noniid_alpha01` seed=123: [FAIL] noniid_consec_strict_spikes: longest streak with |Δ|>0.01 is 3 (rounds 7-9); max allowed consecutive is 2
- **#17** `exp_reverse_adaptive_noniid_alpha01` seed=456: [FAIL] final_communication_mb: mps=1533.125000  cuda=1643.125000  |Δ|=110.000000  (atol=0.01); [FAIL] round_7_avg_loss: mps=1.339824  cuda=1.375425  |Δ|=0.035601  (atol=0.025); [FAIL] round_8_avg_loss: mps=1.372140  cuda=1.327239  |Δ|=0.044901  (atol=0.025); [FAIL] round_12_avg_loss: mps=1.324776  cuda=1.366671  |Δ|=0.041894  (atol=0.025); [FAIL] switch_round_exact: mps=6  cuda=7  (hard match, no tolerance); [FAIL] noniid_consec_strict_spikes: longest streak with |Δ|>0.01 is 3 (rounds 1-3); max allowed consecutive is 2
- **#23** `exp_two_phase_k8_noniid_alpha05` seed=42: [FAIL] final_communication_mb: mps=2578.125000  cuda=1863.125000  |Δ|=715.000000  (atol=0.01); [FAIL] switch_round_exact: mps=None  cuda=9  (hard match, no tolerance)
- **#24** `exp_two_phase_k8_noniid_alpha05` seed=123: [FAIL] final_communication_mb: mps=2578.125000  cuda=1863.125000  |Δ|=715.000000  (atol=0.01); [FAIL] switch_round_exact: mps=None  cuda=9  (hard match, no tolerance)
- **#25** `exp_two_phase_k8_noniid_alpha05` seed=456: [FAIL] final_communication_mb: mps=2578.125000  cuda=1863.125000  |Δ|=715.000000  (atol=0.01); [FAIL] switch_round_exact: mps=None  cuda=9  (hard match, no tolerance)

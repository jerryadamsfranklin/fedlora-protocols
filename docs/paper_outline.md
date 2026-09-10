# Paper Outline: Adaptive Phase-Switching for Communication-Efficient Federated LoRA

## Target venue

Neurocomputing. Single-blind, approximately 5500 word target, no strict page limit.

## Headline claims

1. First measured bidirectional B-only protocol for federated LoRA.
2. Adaptive phase-switching matches or improves on strong fixed-K Two-Phase configurations when threshold τ is set appropriately.
3. Communication savings cluster tightly with other federated methods on downstream accuracy at both TinyLlama and LLaMA-3.2-3B scales under the Stage 5 evaluation protocol.

## Section 1: Introduction

- Federated LLM fine-tuning is constrained by communication.
- Existing federated LoRA methods often report theoretical savings only.
- Contributions: (1) measured bidirectional protocol, (2) adaptive switching, (3) downstream evaluation including HellaSwag, (4) scale validation at LLaMA-3.2-3B.

## Section 2: Related Work

- LoRA, federated LoRA (FedIT, FFA-LoRA, FLoRA, FlexLoRA).
- Communication-efficient federated learning.
- Adaptive aggregation schedules.

## Section 3: Method

- 3.1 Background: federated LoRA aggregation.
- 3.2 Bidirectional B-only protocol with timing diagram.
- 3.3 Adaptive phase-switching (algorithm box, threshold τ explanation).
- 3.4 Implementation details: byte tracking, dtype handling, partition mechanics.

## Section 4: Experiments

- 4.1 Setup: TinyLlama-1.1B-Chat and LLaMA-3.2-3B, Alpaca-3k, 10 clients, 15 rounds.
- 4.2 Communication-quality Pareto (Figure 1): FLoRA, Two-Phase K=10, Two-Phase K=8 with seed aggregates; ReverseAdaptive τ sweep on seed 42 with dashed connector.
- 4.3 Convergence and switching behavior (Figure 2, Figure 3).
- 4.4 Downstream evaluation (Figure 4): TinyLlama panel uses ARC-Easy, BoolQ, HellaSwag (omit MMLU at this scale); LLaMA-3.2-3B panel uses all four benchmarks.
- 4.5 Threshold ablation (Figure 5).
- 4.6 Scale validation (Figure 6): TinyLlama uses three methods with three seeds each (mean ± std); LLaMA uses three methods with one seed each.

## Section 5: Discussion

- Limitations: single principal dataset for training, simulation-only, partial participation not studied.
- Regression relative to base models on some benchmarks appears comparable across federated methods, consistent with instruction-tuning tradeoffs rather than aggregation-specific failure modes for this setup.
- Future work: stronger compression, privacy, deployment.

## Section 6: Conclusion

## Appendix A: Hyperparameters

## Appendix B: Per-seed full results

## Appendix C: Reproducibility checklist

## Numerical claims (support from `analysis/final_results_table.csv` and cited runs)

- Two-Phase K=8 achieves approximately **27.7%** measured round-trip savings versus FLoRA on TinyLlama (Stage 1 bidirectional runs versus `run_*` FLoRA IID totals at round 15) and **26.0%** on LLaMA-3.2-3B (`exp_llama3_two_phase_k8_iid` versus `exp_llama3_flora_iid`, `stage5_llama3_full`).
- ReverseAdaptive achieves approximately **40.5%** savings versus FLoRA on TinyLlama (`stage2_adaptive` versus `run_*` FLoRA) and **30.0%** on LLaMA-3.2-3B (`exp_llama3_reverse_adaptive_iid` versus `exp_llama3_flora_iid`, `stage5_llama3_full`).
- On downstream accuracy at **500 examples per benchmark**, federated methods cluster within about **1.4** percentage points across the reported TinyLlama benchmarks (ARC-Easy, BoolQ, HellaSwag; MMLU omitted at TinyLlama scale as chance-level) and within **1.0** percentage point across MMLU, ARC-Easy, BoolQ, and HellaSwag at LLaMA-3.2-3B for the Stage 5 matrix.
- ReverseAdaptive threshold τ recovers a practical Pareto frontier in threshold-ablation runs: example operating points include τ = 0.001 (later switch, higher cumulative communication), τ = 0.002 (aligns with Two-Phase K=8-style communication on seed 42), and τ = 0.005 (saturated early switch regime). Figure 1 overlays multiple τ values as a single-parameter sweep versus fixed-K Two-Phase points.

# SR-1 Step 3 Inventory

## Structural elements

| Line | Kind | Title/Env | Label | Context |
|------|------|-----------|-------|---------|
| 87 | section | Introduction | `sec:intro` | `\section{Introduction}` |
| 111 | section | Related Work | `sec:related` | `\section{Related Work}` |
| 114 | subsection | LoRA and Parameter-Efficient Fine-Tuning | `` | `\subsection{LoRA and Parameter-Efficient Fine-Tuning}` |
| 117 | subsection | Federated LoRA Aggregation | `` | `\subsection{Federated LoRA Aggregation}` |
| 126 | subsection | Communication-Efficient Federated Learning | `` | `\subsection{Communication-Efficient Federated Learning}` |
| 129 | subsection | Adaptive Aggregation and Scheduling | `sec:method` | `\subsection{Adaptive Aggregation and Scheduling}` |
| 133 | section | Method | `sec:method` | `\section{Method}` |
| 136 | subsection | Background and Notation | `` | `\subsection{Background and Notation}` |
| 141 | subsection | Bidirectional B-Only Protocol | `` | `\subsection{Bidirectional B-Only Protocol}` |
| 156 | algorithm | (env) | `alg:bonly` | `\begin{algorithm}[t]` |
| 178 | subsection | ReverseAdaptive Aggregator | `` | `\subsection{ReverseAdaptive Aggregator}` |
| 187 | algorithm | (env) | `alg:reverseadaptive` | `\begin{algorithm}[t]` |
| 213 | subsection | Implementation Details | `sec:implementation` | `\subsection{Implementation Details}` |
| 226 | section | Experiments and Results | `sec:experiments` | `\section{Experiments and Results}` |
| 229 | subsection | Setup | `sec:setup` | `\subsection{Setup}` |
| 233 | table | (env) | `tab:setup` | `\begin{table}[tbp]` |
| 277 | subsection | Held-Out Instruction-Following Metric | `sec:heldout` | `\subsection{Held-Out Instruction-Following Metric}` |
| 281 | subsection | The Communication-Quality Frontier | `sec:frontier` | `\subsection{The Communication-Quality Frontier}` |
| 285 | table | (env) | `tab:frontier` | `\begin{table}[tbp]` |
| 305 | figure | (env) | `fig:frontier` | `\begin{figure}[tbp]` |
| 324 | subsection | Replication on Dolly-15k | `sec:dolly` | `\subsection{Replication on Dolly-15k}` |
| 328 | table | (env) | `tab:dolly` | `\begin{table}[tbp]` |
| 351 | subsection | Convergence and Switching Behavior | `fig:convergence` | `\subsection{Convergence and Switching Behavior}` |
| 352 | figure | (env) | `fig:convergence` | `\begin{figure}[tbp]` |
| 364 | subsection | Downstream Evaluation | `` | `\subsection{Downstream Evaluation}` |
| 369 | table | (env) | `tab:downstream_tiny` | `\begin{table}[tbp]` |
| 391 | table | (env) | `tab:downstream_llama` | `\begin{table}[bp]` |
| 409 | subsection | Threshold Ablation | `sec:ablation` | `\subsection{Threshold Ablation}` |
| 415 | table | (env) | `tab:ablation` | `\begin{table}[tbp]` |
| 432 | figure | (env) | `fig:ablation` | `\begin{figure}[tbp]` |
| 439 | subsection | Scale Validation | `` | `\subsection{Scale Validation}` |
| 444 | table | (env) | `tab:llama_iid` | `\begin{table}[tbp]` |
| 461 | figure | (env) | `fig:scale` | `\begin{figure}[tbp]` |
| 474 | section | Discussion | `sec:discussion` | `\section{Discussion}` |
| 477 | subsection | Implications for Federated LLM Deployment | `sec:choosing` | `\subsection{Implications for Federated LLM Deployment}` |
| 480 | subsection | Choosing an Operating Point | `sec:choosing` | `\subsection{Choosing an Operating Point}` |
| 493 | subsection | Measured Bytes and Theoretical Parameter Counts | `sec:measuredbytes` | `\subsection{Measured Bytes and Theoretical Parameter Counts}` |
| 505 | subsection | Zero-Shot Benchmark Behavior and the Training Objective | `sec:baseregression` | `\subsection{Zero-Shot Benchmark Behavior and the Training Objective}` |
| 515 | section | Limitations | `sec:limitations` | `\section{Limitations}` |
| 542 | section | Conclusion | `sec:conclusion` | `\section{Conclusion}` |
| 587 | section | Experimental Details | `app:hyper` | `\section{Experimental Details}` |
| 591 | subsection | Reproducibility Checklist | `app:repro` | `\subsection{Reproducibility Checklist}` |
| 603 | subsection | Disabling the Switch | `app:noswitch` | `\subsection{Disabling the Switch}` |
| 610 | section | Per-Seed Full Results | `app:perseed` | `\section{Per-Seed Full Results}` |
| 614 | table | (env) | `` | `\begin{table}[tbp]` |
| 647 | table | (env) | `tab:b2` | `\begin{table}[tbp]` |
| 669 | table | (env) | `tab:b3` | `\begin{table}[tbp]` |
| 692 | table | (env) | `tab:b4` | `\begin{table}[tbp]` |
| 715 | section | Statistical Tests | `app:stats` | `\section{Statistical Tests}` |
| 723 | table | (env) | `tab:c1` | `\begin{table}[tbp]` |
| 754 | section | Cross-Backend Verification | `app:crossbackend` | `\section{Cross-Backend Verification}` |
| 763 | table | (env) | `tab:d1` | `\begin{table}[tbp]` |
| 785 | table | (env) | `` | `\begin{table}[tbp]` |

## Cross-references

| Line | Label | Context (80 chars) |
|------|-------|-------------------|
| 108 | `app:repro` | `blicly released; see Appendix~\ref{app:repro}.` |
| 122 | `sec:implementation` | `he byte accounting of Section~\ref{sec:implementation}, sharing $A$ instead of $` |
| 139 | `sec:implementation` | `B$ to $A$ parameters; Section~\ref{sec:implementation} derives the exact fractio` |
| 154 | `alg:bonly` | `Algorithm~\ref{alg:bonly} specifies the server-side decision` |
| 179 | `sec:choosing` | `thout retuning, which Section~\ref{sec:choosing} examines. A complementary stabi` |
| 181 | `sec:ablation` | `e round after warmup. Section~\ref{sec:ablation} characterizes this tradeoff emp` |
| 185 | `alg:reverseadaptive` | `Algorithm~\ref{alg:reverseadaptive} specifies the aggregator` |
| 224 | `sec:limitations` | ` than task semantics; Section~\ref{sec:limitations} notes that semantic heteroge` |
| 231 | `tab:setup` | `Table~\ref{tab:setup} summarizes the experimental config` |
| 231 | `app:crossbackend` | `rison is reported in Appendix~\ref{app:crossbackend}. The Dolly-15k replication ` |
| 273 | `sec:heldout` | `l reference values of Section~\ref{sec:heldout}.` |
| 275 | `app:crossbackend` | ` switching boundary (Appendix~\ref{app:crossbackend}). Loss values differ across` |
| 275 | `app:crossbackend` | ` switching boundary (Appendix~\ref{app:crossbackend}). Loss values differ across` |
| 275 | `tab:frontier` | `e five-method frontier (Table~\ref{tab:frontier}), the Dolly-15k replication (Ta` |
| 275 | `tab:dolly` | ` Dolly-15k replication (Table~\ref{tab:dolly}), and all statistical tests use th` |
| 275 | `sec:ablation` | `e threshold ablation (Section~\ref{sec:ablation}) and the LLaMA-3.2-3B results u` |
| 275 | `app:crossbackend` | ` switching boundary (Appendix~\ref{app:crossbackend}). Loss values differ across` |
| 279 | `sec:baseregression` | `points at this scale (Section~\ref{sec:baseregression}). Quality is therefore al` |
| 279 | `sec:setup` | `t format described in Section~\ref{sec:setup}. Each adapted checkpoint is scored` |
| 279 | `sec:dolly` | `dataset comparison in Section~\ref{sec:dolly} therefore compares between-method ` |
| 283 | `tab:frontier` | `Table~\ref{tab:frontier} and Fig.~\ref{fig:frontier} pla` |
| 283 | `fig:frontier` | `e~\ref{tab:frontier} and Fig.~\ref{fig:frontier} place five federated LoRA proto` |
| 289 | `sec:heldout` | ` 500-example slice of Section~\ref{sec:heldout}; more negative is better. Differ` |
| 289 | `app:stats` | `n both metrics is in Appendix~\ref{app:stats}. Methods: FLoRA \cite{wang2024flor` |
| 308 | `sec:heldout` | `mple slice defined in Section~\ref{sec:heldout}; more negative is better. FLoRA ` |
| 308 | `sec:measuredbytes` | `t preceding it, which Section~\ref{sec:measuredbytes} quantifies}` |
| 312 | `tab:frontier` | `Three features of Table~\ref{tab:frontier} bear on the interpretation. The` |
| 314 | `sec:implementation` | `ents trained against (Section~\ref{sec:implementation}). Learning $A$ through a ` |
| 318 | `app:stats` | `ower at three seeds (Appendix~\ref{app:stats}), so effect sizes should be read a` |
| 318 | `tab:frontier` | `orrection is applied in Table~\ref{tab:frontier}; Appendix~\ref{app:stats} repor` |
| 318 | `app:stats` | `ower at three seeds (Appendix~\ref{app:stats}), so effect sizes should be read a` |
| 322 | `app:crossbackend` | `t backend-invariant; Appendix~\ref{app:crossbackend} reports that both tight-$\t` |
| 326 | `tab:dolly` | `Table~\ref{tab:dolly} repeats the three-method compariso` |
| 332 | `sec:method` | `aptive are defined in Section~\ref{sec:method}. The MB column is total round-tri` |
| 345 | `tab:dolly` | `608 on the Dolly slice. Table~\ref{tab:dolly} reports the gap to each dataset's ` |
| 359 | `fig:convergence` | `Fig.~\ref{fig:convergence} shows mean $\pm$ std loss tr` |
| 361 | `app:perseed` | `A per-seed values in Appendix~\ref{app:perseed} differ, as expected across backe` |
| 362 | `app:noswitch` | `r reasons set out in Appendix~\ref{app:noswitch}.` |
| 365 | `tab:downstream_tiny` | `Tables~\ref{tab:downstream_tiny}--\ref{tab:downstream_lla` |
| 365 | `tab:downstream_llama` | `es~\ref{tab:downstream_tiny}--\ref{tab:downstream_llama} present downstream resu` |
| 367 | `tab:downstream_tiny` | `At TinyLlama-1.1B (Table~\ref{tab:downstream_tiny}), all federated methods ` |
| 367 | `sec:baseregression` | `ve to the base model (Section~\ref{sec:baseregression}).` |
| 389 | `sec:baseregression` | `t regress as at 1.1B (Section~\ref{sec:baseregression}): the LLaMA base achieves` |
| 411 | `tab:ablation` | `Table~\ref{tab:ablation} and Fig.~\ref{fig:ablation} cha` |
| 411 | `fig:ablation` | `e~\ref{tab:ablation} and Fig.~\ref{fig:ablation} characterize how $\tau$ control` |
| 435 | `tab:frontier` | `ith the CUDA results in Table~\ref{tab:frontier}, and the $\tau$=0.001 and $\tau` |
| 435 | `app:crossbackend` | `und earlier on CUDA (Appendix~\ref{app:crossbackend})}` |
| 440 | `tab:llama_iid` | `Table~\ref{tab:llama_iid} and Fig.~\ref{fig:scale} confi` |
| 440 | `fig:scale` | `~\ref{tab:llama_iid} and Fig.~\ref{fig:scale} confirm that the Pareto structure ` |
| 478 | `sec:frontier` | `, and the frontier in Section~\ref{sec:frontier} is intended to let a practition` |
| 484 | `app:crossbackend` | `d CUDA backends (see Appendix~\ref{app:crossbackend}), so the cross-scale compar` |
| 486 | `app:crossbackend` | ` round 7 instead, as Appendix~\ref{app:crossbackend} reports. At $\tau{=}0.01$ a` |
| 486 | `sec:ablation` | `h rounds of 9 and 11 (Section~\ref{sec:ablation}). Across seeds in IID training,` |
| 486 | `sec:limitations` | `fer in code revision (Section~\ref{sec:limitations}), so the variation across th` |
| 501 | `app:perseed` | `articipation runs in Appendix~\ref{app:perseed} scale off the lattice in proport` |
| 501 | `sec:frontier` | `K$ total robustly, as Section~\ref{sec:frontier} notes.` |
| 503 | `sec:implementation` | `m the initialization (Section~\ref{sec:implementation}). An implementation faith` |
| 503 | `sec:limitations` | `tes, and is untested (Section~\ref{sec:limitations}).` |
| 511 | `tab:frontier` | `ee-seed CUDA results in Table~\ref{tab:frontier}, so the spreads carry no varian` |
| 520 | `tab:dolly` | `ated on Alpaca only, so Table~\ref{tab:dolly} compares three protocols where Tab` |
| 520 | `tab:frontier` | `s three protocols where Table~\ref{tab:frontier} compares five. The cross-datase` |
| 532 | `sec:baseregression` | `s at TinyLlama scale (Section~\ref{sec:baseregression}), so conclusions about re` |
| 534 | `sec:implementation` | `lity is not measured (Section~\ref{sec:implementation}). Freezing the global $A$` |
| 536 | `tab:frontier` | `FLoRA and FedIT rows of Table~\ref{tab:frontier} sit on either side of that boun` |
| 574 | `tab:downstream_tiny` | `-1.1B runs evaluated in Table~\ref{tab:downstream_tiny} are included; other adap` |
| 589 | `tab:setup` | `ngth is 256 tokens. See Table~\ref{tab:setup} for the complete configuration inc` |
| 597 | `tab:setup` | `IID seeds are listed in Table~\ref{tab:setup}. The $\alpha{=}0.1$ partition-sens` |
| 597 | `app:perseed` | `nsitivity results of Appendix~\ref{app:perseed}, alongside the other per-seed re` |
| 599 | `app:crossbackend` | `the comparison is in Appendix~\ref{app:crossbackend}. The Dolly-15k replication ` |
| 601 | `tab:downstream_tiny` | `-1.1B runs evaluated in Table~\ref{tab:downstream_tiny}.` |
| 605 | `app:crossbackend` | `y and is reported in Appendix~\ref{app:crossbackend}.` |
| 612 | `tab:b1` | `Tables~\ref{tab:b1} to~\ref{tab:b4} give per-seed results` |
| 612 | `tab:b4` | `Tables~\ref{tab:b1} to~\ref{tab:b4} give per-seed results and are labeled` |
| 612 | `tab:llama_iid` | `sults use MPS, matching Table~\ref{tab:llama_iid}; and the partition-sensitivity` |
| 620 | `tab:frontier` | `s, CUDA corpus, backing Table~\ref{tab:frontier}. Communication is deterministic` |
| 645 | `tab:b1` | `s 10^{-3}$ under IID in Table~\ref{tab:b1} to $18$ to $25 \times 10^{-3}$ at $\a` |
| 645 | `tab:b2` | `ad at $\alpha{=}0.1$ in Table~\ref{tab:b2} is wide.` |
| 651 | `tab:b1` | `S seed set is retained; Table~\ref{tab:b1} follows the CUDA corpus because it ba` |
| 651 | `app:crossbackend` | `ackend verification (Appendix~\ref{app:crossbackend}) covers seeds 42, 123, and ` |
| 673 | `tab:llama_iid` | `ts, MPS corpus, backing Table~\ref{tab:llama_iid}. ReverseAdaptive uses $\tau{=}` |
| 696 | `tab:dolly` | `s, CUDA corpus, backing Table~\ref{tab:dolly}. Held-out $\Delta$ is tuned minus ` |
| 717 | `tab:c1` | `Table~\ref{tab:c1} reports paired $t$-tests over three s` |
| 717 | `tab:frontier` | `e CUDA corpus, matching Table~\ref{tab:frontier}; LLaMA-3.2-3B results use the M` |
| 719 | `sec:discussion` | `e frontier framing of Section~\ref{sec:discussion}; the load-bearing comparison ` |
| 757 | `tab:d1` | `ly re-executed on CUDA. Table~\ref{tab:d1} summarizes the comparison; Table~\ref` |
| 757 | `tab:d2` | `marizes the comparison; Table~\ref{tab:d2} details every disagreement.` |
| 805 | `sec:measuredbytes` | `MB switch-round step (Section~\ref{sec:measuredbytes}), so Table~\ref{tab:d2} li` |
| 805 | `tab:d2` | `sec:measuredbytes}), so Table~\ref{tab:d2} lists adjacent lattice points. The ba` |
| 807 | `sec:choosing` | ` cross-scale claim in Section~\ref{sec:choosing}.` |
| 811 | `tab:frontier` | `FLoRA and FedIT rows of Table~\ref{tab:frontier} differ by 0.0006 across that bo` |

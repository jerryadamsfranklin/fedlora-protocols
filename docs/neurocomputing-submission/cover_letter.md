# Cover Letter

To the Editors, *Neurocomputing*

**Manuscript:** Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning
**Author:** Jerry Adams Franklin, Independent Researcher

---

Dear Editors,

Please consider the enclosed manuscript for publication in *Neurocomputing*.

The paper addresses a measurement problem in federated fine-tuning of large language models. Existing work on federated low-rank adaptation reports communication savings as parameter-count ratios rather than as bytes actually transmitted. Two consequences follow, and neither is visible in a parameter ratio. Protocols that change aggregation mode mid-training incur an asymmetric transition cost that ratio-based accounting omits entirely. And the fraction of a low-rank adapter that a protocol can decline to transmit is fixed by the model's grouped-query attention configuration, so a savings figure derived for one architecture does not carry to another.

The primary contribution is therefore empirical rather than algorithmic. We instrument a bidirectional federated LoRA protocol at the transport layer, counting the bytes on the wire rather than estimating them from parameter counts, and use that instrumentation to place five published protocols on a single measured communication-quality frontier at two model scales and on two instruction-following datasets. The frontier turns out to have a knee: the first 40.5 percentage points of communication savings cost 0.0063 in held-out instruction-following loss, while the next 21.3 points cost roughly five times more per point. That structure is the paper's central finding, and it is not derivable from any parameter-count analysis.

A secondary contribution operationalizes it. ReverseAdaptive is a schedule that locates the knee by monitoring the relative per-round improvement in global training loss against a dimensionless threshold, rather than by fixing a phase boundary in advance. Because the criterion is scale-free, we transferred a single threshold value from a 1.1B-parameter model to a 3B-parameter model without retuning; the switch round matched exactly across two hardware backends.

We also report results that do not favor the proposed method, because they bound what the measurements support. The four zero-shot benchmarks commonly used in this literature do not discriminate between aggregation protocols at the smaller scale, where every fine-tuned checkpoint scores below the base model; quality conclusions therefore rest on held-out instruction-following loss alone. At 3B, the adaptive and hand-tuned configurations differ by an amount three seeds cannot resolve, and we report this as an effect size rather than as evidence of equivalence. The 55 MB transition cost we measure is a property of our implementation's frozen-parameter harvest rule rather than of phase-switching protocols in general, and we identify the untested alternative that would eliminate it. We also disclose that our own measurement charges the strongest baseline 55 MB it should not pay, and show that correcting it moves the headline ratio from 5.5 to 5.0 while leaving the conclusion unchanged.

All experiments were produced on a single consumer workstation, approximately 285 hours of compute with no institutional cluster, and subsequently reproduced on rented GPU hardware. The cross-backend comparison is reported in full, including every disagreement.

Code, configuration files, analysis data, and scripts sufficient to regenerate every figure and table are publicly released. Adapter checkpoints are not released; the reproduction scripts are sufficient without them.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. I have no competing interests to declare. As sole author I am responsible for all aspects of the work.

The author used Claude (Anthropic) for grammar and language editing only. All technical content, analysis, and interpretations are the author's own.

Thank you for your consideration.

Sincerely,

Jerry Adams Franklin
Independent Researcher
jerry.adamsf@gmail.com
ORCID: 0009-0006-8470-8349

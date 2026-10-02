To: Professor Yaser Jararweh, Editor-in-Chief
Cluster Computing

Manuscript: A Measured Communication-Quality Frontier for Federated LoRA Fine-Tuning with Adaptive Phase-Switching
Author: Jerry Adams Franklin, Independent Researcher

Dear Professor Jararweh,

Please consider the enclosed manuscript for publication in Cluster Computing.

The paper measures per-round upload and download bytes for five federated LoRA protocols, three of them from prior work, and places them on a single communication-quality frontier scored by held-out instruction-following loss. On TinyLlama-1.1B with Alpaca the frontier has a knee: the first 40.5 percentage points of communication savings relative to FLoRA cost 0.0063 in held-out loss, while further savings toward FFA-LoRA cost roughly five times more per point. ReverseAdaptive, which learns both LoRA factors before freezing one once relative training-loss improvement falls below a dimensionless threshold, sits at that knee, and the same threshold transfers to LLaMA-3.2-3B without retuning.

The work fits Cluster Computing's scope because federated fine-tuning is a distributed-systems problem whose dominant cost is communication across a client network. The contribution is measurement and protocol comparison under a shared byte-accounting path, including the asymmetric transition round incurred when a protocol changes aggregation mode mid-training, rather than a new distributed algorithm in isolation.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. An earlier version is available as arXiv:2609.13512. I have no competing interests to declare. As sole author I am responsible for all aspects of the work.

Thank you for your consideration.

Sincerely,

Jerry Adams Franklin
Independent Researcher
jerry.adamsf@gmail.com
ORCID: 0009-0006-8470-8349

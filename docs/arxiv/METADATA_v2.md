# arXiv:2609.13512 replacement metadata (v2)

Copy-paste blocks for the arXiv Replace form.

## Action
Replace on arXiv:2609.13512 (not a new submission).

## Title
A Measured Communication-Quality Frontier for Federated LoRA Fine-Tuning with Adaptive Phase-Switching

## Abstract
(1036 characters; limit 1920)

Federated fine-tuning of large language models with low-rank adaptation (LoRA) reduces the number of trainable parameters, but communication remains the dominant cost, and protocols are usually compared by parameter-count ratios rather than by measured bytes. This paper measures per-round upload and download bytes for five federated LoRA protocols, three of them from prior work, and places them on a single communication-quality frontier scored by held-out instruction-following loss. The frontier has a knee. ReverseAdaptive, which learns both LoRA factors before freezing one once the relative improvement in training loss falls below a dimensionless threshold, sits at that knee: it cuts measured round-trip communication by 40.5% relative to FLoRA at a held-out loss cost of 0.0063, and beats FFA-LoRA, which freezes that factor at initialization, by 0.0182 in held-out loss, more than twenty times the largest per-method seed standard deviation. The same threshold carries to LLaMA-3.2-3B without retuning, where it saves 30.0%.

## Comments
20 pages, 4 figures. v2: revised and retitled version with corrected text, figures, and appendices. Code and data: https://doi.org/10.5281/zenodo.23111915

## Categories
cs.AI primary; cs.LG, cs.DC cross-lists (unchanged).

## ACM classes
C.2.4; C.4; I.2.6; I.2.7; I.2.11 (v1 values; confirm on the v1 abstract page).

## Licence
Same as v1 (read it on https://arxiv.org/abs/2609.13512; a replacement cannot be more restrictive).

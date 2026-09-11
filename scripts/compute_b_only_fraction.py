#!/usr/bin/env python3
"""
Compute LoRA B-only communication fractions.

For each attention projection (q/k/v/o) with LoRA rank r:
  A params = r * d_in
  B params = d_out * r
  B-only fraction = d_out / (d_in + d_out)   (rank cancels)

Uses published Llama* config geometry (no model download required).
Target modules match config/base_config_4layers.yaml and base_config_llama3_3b.yaml:
q_proj, k_proj, v_proj, o_proj.

The B-only fraction is nearly scale-invariant across TinyLlama-1.1B, LLaMA-3.2-3B,
and LLaMA-3.1-8B geometries (within a few percentage points).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LlamaAttnGeometry:
    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    # HuggingFace / paper model ids (for logging only)
    hf_id: str


# Canonical shapes for the model scales used in the paper.
# TinyLlama: HF TinyLlama-1.1B-Chat-v1.0 config.json
# LLaMA-3.2-3B / LLaMA-3.1-8B: Meta Llama GQA configs (same target modules).
MODELS: List[LlamaAttnGeometry] = [
    LlamaAttnGeometry(
        name="TinyLlama-1.1B",
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=22,
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    LlamaAttnGeometry(
        name="LLaMA-3.2-3B",
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,
        num_hidden_layers=28,
        hf_id="meta-llama/Llama-3.2-3B",
    ),
    LlamaAttnGeometry(
        name="LLaMA-3.1-8B",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        hf_id="meta-llama/Meta-Llama-3.1-8B",
    ),
]

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_RANK = 16


def layer_dims(geom: LlamaAttnGeometry) -> Dict[str, Tuple[int, int]]:
    """Return (d_in, d_out) for each LoRA target module."""
    h = geom.hidden_size
    head_dim = h // geom.num_attention_heads
    kv_out = geom.num_key_value_heads * head_dim
    return {
        "q_proj": (h, h),
        "k_proj": (h, kv_out),
        "v_proj": (h, kv_out),
        "o_proj": (h, h),
    }


def count_lora_ab(
    geom: LlamaAttnGeometry, rank: int = DEFAULT_RANK
) -> Tuple[int, int, Dict[str, float]]:
    """Total A/B LoRA params across all layers; per-module B fractions."""
    dims = layer_dims(geom)
    a_total = 0
    b_total = 0
    per_module_frac: Dict[str, float] = {}
    for name in TARGET_MODULES:
        d_in, d_out = dims[name]
        a = rank * d_in
        b = d_out * rank
        a_total += a * geom.num_hidden_layers
        b_total += b * geom.num_hidden_layers
        per_module_frac[name] = d_out / (d_in + d_out)
    return a_total, b_total, per_module_frac


def main() -> None:
    print("B-only communication fractions")
    print(f"LoRA rank r={DEFAULT_RANK} (cancels in fraction); targets={TARGET_MODULES}")
    print()

    rows = []
    for geom in MODELS:
        a, b, per_mod = count_lora_ab(geom)
        frac = b / (a + b)
        rows.append((geom, a, b, frac, per_mod))
        print(f"{geom.name} ({geom.hf_id})")
        print(
            f"  layers={geom.num_hidden_layers}  hidden={geom.hidden_size}  "
            f"heads={geom.num_attention_heads}/{geom.num_key_value_heads} GQA"
        )
        print(f"  LoRA A params: {a:,}")
        print(f"  LoRA B params: {b:,}")
        print(f"  B-only fraction: {frac:.4f}  ({100.0 * frac:.2f}%)")
        print(
            "  per-module B/(A+B): "
            + ", ".join(f"{k}={v:.3f}" for k, v in per_mod.items())
        )
        print()

    fracs = [r[3] for r in rows]
    spread_pp = 100.0 * (max(fracs) - min(fracs))
    print("--- Cross-scale check ---")
    print(
        "  "
        + ", ".join(
            f"{geom.name}={100.0 * frac:.2f}%" for geom, _, _, frac, _ in rows
        )
    )
    print(f"  max−min spread: {spread_pp:.2f} percentage points")
    if spread_pp <= 5.0:
        print("  PASS: fractions are within a few percentage points (≤ 5 pp).")
    else:
        print(
            "  FAIL: spread exceeds 5 pp — check that LoRA targets match across scales."
        )
        raise SystemExit(1)

    # Paper-reported anchors for the two existing scales
    paper = {"TinyLlama-1.1B": 0.36, "LLaMA-3.2-3B": 0.40}
    print()
    print("--- vs paper draft anchors (~36% TinyLlama, ~40% LLaMA-3.2-3B) ---")
    for geom, _, _, frac, _ in rows:
        if geom.name in paper:
            ok = abs(frac - paper[geom.name]) < 1e-6
            status = "MATCH" if ok else "MISMATCH"
            print(
                f"  {geom.name}: computed {100.0 * frac:.2f}% vs paper "
                f"{100.0 * paper[geom.name]:.2f}%  [{status}]"
            )
    print(
        f"  LLaMA-3.1-8B (new): {100.0 * fracs[2]:.2f}%  "
        f"(between TinyLlama and 3.2-3B; consistent GQA shape)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_numbers.py -- check every reported number in the manuscript against the CSVs.

Design note: this deliberately does NOT parse body.tex. LaTeX parsing produces
false failures that cost more time than they save. Instead each claim is a
curated assertion carrying (a) how to recompute it from a CSV, (b) the value the
manuscript states, and (c) where in the manuscript it appears. A failure names
the exact location to edit.

Adding a claim is one entry in ASSERTIONS.

Usage:
    python verify_numbers.py --analysis-dir analysis
    python verify_numbers.py --analysis-dir analysis --filter abstract
    python verify_numbers.py --analysis-dir analysis --show-computed
"""

from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Constants that encode protocol structure. These are claims too.
# --------------------------------------------------------------------------

FULL_PAYLOAD_BYTES_TINYLLAMA = 9_011_200   # q/v, r=16, 22 layers, float32, per client per direction
BONLY_PAYLOAD_BYTES_TINYLLAMA = 3_244_032  # B only, same config
N_CLIENTS = 10
N_ROUNDS = 15

# TinyLlama step function: total = STEP * (2s - 1) + ALL_BONLY, s = switch round.
TL_STEP = 55.0
TL_ALL_BONLY = 928.125
# LLaMA-3.2-3B equivalent.
L3_STEP = 52.5
L3_ALL_BONLY = 1050.0


# For a switching method with switch round s, full-payload directions total
# (2s - 1): uploads are full for rounds 1..s, broadcasts for rounds 1..s-1. A
# non-switching method has all 2*N_ROUNDS directions full. FLoRA is therefore
# NOT the s = N+1 point of the switching family; it is the separate all-full
# endpoint. Conflating the two overstates FLoRA by one step.

def tl_total(s: int) -> float:
    return TL_STEP * (2 * s - 1) + TL_ALL_BONLY


def tl_no_switch() -> float:
    return TL_STEP * (2 * N_ROUNDS) + TL_ALL_BONLY


def l3_total(s: int) -> float:
    return L3_STEP * (2 * s - 1) + L3_ALL_BONLY


def l3_no_switch() -> float:
    return L3_STEP * (2 * N_ROUNDS) + L3_ALL_BONLY


# --------------------------------------------------------------------------
# Loaders. Every filter here is a documented decision, not a convenience.
# --------------------------------------------------------------------------

class Data:
    def __init__(self, d: str):
        self.dir = d
        self.runs = self._csv("qv_only_cuda_per_seed_runs.csv")
        self.final = self._csv("final_results_table.csv")
        hold = []
        for name in ("neuro_part12_qvonly_holdout_strict24.csv",
                     "neuro_existing_alpaca_holdout_phase1_only.csv"):
            hold.append(self._csv(name))
        self.hold = pd.concat(hold, ignore_index=True)

    def _csv(self, name: str) -> pd.DataFrame:
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            raise SystemExit(f"missing required CSV: {p}")
        return pd.read_csv(p)

    # -- communication -----------------------------------------------------

    def comm(self, exp: str) -> float:
        """Total communication MB. Asserts seed-invariance for deterministic protocols."""
        v = self.runs.loc[self.runs.exp_name == exp, "total_communication_mb"]
        if v.empty:
            raise KeyError(f"no communication rows for {exp}")
        if v.max() - v.min() > 1e-6:
            raise ValueError(
                f"{exp}: communication varies across seeds "
                f"({v.min()}..{v.max()}); expected protocol-deterministic"
            )
        return float(v.iloc[0])

    def broadcast_bytes(self, exp: str) -> int:
        v = self.runs.loc[self.runs.exp_name == exp, "broadcast_bytes_per_client"].unique()
        if len(v) != 1:
            raise ValueError(f"{exp}: broadcast_bytes_per_client not unique: {v}")
        return int(v[0])

    # -- quality -----------------------------------------------------------

    def final_loss(self, exp: str) -> tuple[float, float, int]:
        v = self.runs.loc[self.runs.exp_name == exp, "final_avg_loss"].tolist()
        if not v:
            raise KeyError(f"no loss rows for {exp}")
        return st.mean(v), st.stdev(v), len(v)

    def heldout(self, exp: str) -> tuple[float, float, int]:
        v = self.hold.loc[self.hold.exp_name == exp, "delta_loss_tuned_minus_base"].tolist()
        if not v:
            raise KeyError(f"no holdout rows for {exp}")
        return st.mean(v), st.stdev(v), len(v)

    def paired_p(self, exp_a: str, exp_b: str, metric: str) -> float:
        """Paired t-test on seed-matched runs. metric in {'loss', 'heldout'}."""
        if metric == "loss":
            src, col = self.runs, "final_avg_loss"
        else:
            src, col = self.hold, "delta_loss_tuned_minus_base"
        a = src.loc[src.exp_name == exp_a].set_index("seed")[col]
        b = src.loc[src.exp_name == exp_b].set_index("seed")[col]
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) < 2:
            raise ValueError(f"{exp_a} vs {exp_b}: only {len(seeds)} shared seeds")
        return float(stats.ttest_rel(a.loc[seeds], b.loc[seeds]).pvalue)

    # -- LLaMA-3.2-3B ------------------------------------------------------
    # NOTE: final_results_table.csv contains single-round memory smoke tests for
    # 3B FLoRA (35.0 and 70.0 MB totals). A naive groupby yields 902 MB instead
    # of 2625. The total_mb filter below is load-bearing, not cosmetic.

    def l3(self, method: str) -> pd.DataFrame:
        d = self.final[(self.final.scale == "llama3_3b") & (self.final.setting == "iid")]
        d = d[d.method == method]
        if method == "flora":
            d = d[d.total_mb == l3_no_switch()]  # excludes 1-round smoke tests
        return d

    def l3_comm(self, method: str) -> tuple[float, float, int]:
        v = self.l3(method).total_mb.tolist()
        return st.mean(v), st.stdev(v), len(v)

    def l3_savings_pct(self) -> tuple[float, float]:
        base = st.mean(self.l3("flora").total_mb.tolist())
        v = [(base - m) / base * 100 for m in self.l3("reverse_adaptive").total_mb]
        return st.mean(v), st.stdev(v)

    def l3_paired_p(self, m_a: str, m_b: str) -> float:
        a = self.l3(m_a).set_index("seed").final_loss
        b = self.l3(m_b).set_index("seed").final_loss
        seeds = sorted(set(a.index) & set(b.index))
        return float(stats.ttest_rel(a.loc[seeds], b.loc[seeds]).pvalue)


# --------------------------------------------------------------------------
# Assertions. (id, location, expected, tol, fn)
# --------------------------------------------------------------------------

FLORA = "exp_flora_iid"
FEDIT = "exp_fedit_iid"
TP8 = "exp_two_phase_k8"
RA = "exp_reverse_adaptive_iid"
FFA = "exp_ffa_lora_iid"

D_FLORA = "exp_dolly_flora_iid"
D_TP8 = "exp_dolly_two_phase_k8_iid"
D_RA = "exp_dolly_reverse_adaptive_iid"

FRONTIER = [(FLORA, "FLoRA"), (FEDIT, "FedIT"), (TP8, "Two-Phase K=8"),
            (RA, "ReverseAdaptive"), (FFA, "FFA-LoRA")]


def build(d: Data) -> list[tuple]:
    A: list[tuple] = []

    def add(aid, loc, expected, tol, fn):
        A.append((aid, loc, expected, tol, fn))

    base_comm = d.comm(FLORA)

    # -- T-N1: communication ------------------------------------------------
    for exp, label in FRONTIER:
        expected = {FLORA: 2578.125, FEDIT: 2578.125, TP8: 1863.125,
                    RA: 1533.125, FFA: 983.125}[exp]
        add(f"comm.{label}", "T-N1 col 2", expected, 1e-6,
            lambda e=exp: d.comm(e))

    # -- T-N1: savings ------------------------------------------------------
    for exp, label, pct in [(TP8, "Two-Phase K=8", 27.7), (RA, "ReverseAdaptive", 40.5),
                            (FFA, "FFA-LoRA", 61.9)]:
        add(f"savings.{label}", "T-N1 col 3, Abstract", pct, 0.05,
            lambda e=exp: (base_comm - d.comm(e)) / base_comm * 100)

    # -- T-N1: final loss ---------------------------------------------------
    for exp, label, mean in [(FLORA, "FLoRA", 1.2608), (FEDIT, "FedIT", 1.2602),
                             (TP8, "Two-Phase K=8", 1.2705), (RA, "ReverseAdaptive", 1.2749),
                             (FFA, "FFA-LoRA", 1.3031)]:
        add(f"loss.{label}", "T-N1 col 4", mean, 5e-5, lambda e=exp: d.final_loss(e)[0])
        add(f"loss.{label}.n", "T-N1 caption (3 seeds)", 3, 0, lambda e=exp: d.final_loss(e)[2])

    # -- T-N1: held-out -----------------------------------------------------
    for exp, label, mean in [(FLORA, "FLoRA", -0.5992), (FEDIT, "FedIT", -0.5990),
                             (TP8, "Two-Phase K=8", -0.5958), (RA, "ReverseAdaptive", -0.5929),
                             (FFA, "FFA-LoRA", -0.5746)]:
        add(f"heldout.{label}", "T-N1 col 5", mean, 5e-5, lambda e=exp: d.heldout(e)[0])
        add(f"heldout.{label}.n", "T-N1 caption (3 seeds)", 3, 0, lambda e=exp: d.heldout(e)[2])

    # -- Abstract / Section 5.2 derived quantities --------------------------
    add("gap.RA_vs_FLoRA.heldout", "Abstract, Sec 5.2", 0.006306, 5e-6,
        lambda: d.heldout(RA)[0] - d.heldout(FLORA)[0])
    add("gap.FFA_vs_RA.heldout", "Abstract, Sec 5.2, T-N1 discussion", 0.018220, 5e-6,
        lambda: d.heldout(FFA)[0] - d.heldout(RA)[0])
    add("gap.FFA_vs_RA.loss", "Sec 4.2 (G5 point 3)", 0.028246, 5e-6,
        lambda: d.final_loss(FFA)[0] - d.final_loss(RA)[0])
    add("gap.FedIT_vs_FLoRA.loss", "Sec 4.2 (G5 point 4, within 0.0006)", 0.000566, 5e-6,
        lambda: abs(d.final_loss(FEDIT)[0] - d.final_loss(FLORA)[0]))
    add("segment.RA_to_FFA.pp", "Abstract, Sec 5.2 (21.3 pp)", 21.34, 0.05,
        lambda: (d.comm(RA) - d.comm(FFA)) / base_comm * 100)

    def marginal_ratio():
        s1 = (base_comm - d.comm(RA)) / base_comm * 100
        q1 = d.heldout(RA)[0] - d.heldout(FLORA)[0]
        s2 = (d.comm(RA) - d.comm(FFA)) / base_comm * 100
        q2 = d.heldout(FFA)[0] - d.heldout(RA)[0]
        return (q2 / s2) / (q1 / s1)

    add("marginal.ratio", "Abstract, Sec 5.2, Conclusion (5.5x)", 5.5, 0.05, marginal_ratio)

    # -- Appendix C: paired tests ------------------------------------------
    for a, b, metric, label, p in [
        (TP8, FLORA, "loss", "TP8_vs_FLoRA.loss", 4.18e-5),
        (RA, FLORA, "loss", "RA_vs_FLoRA.loss", 1.09e-5),
        (RA, TP8, "loss", "RA_vs_TP8.loss", 3.40e-4),
        (FFA, RA, "loss", "FFA_vs_RA.loss", 4.40e-4),
        (FEDIT, FLORA, "loss", "FedIT_vs_FLoRA.loss", 0.588),
        (TP8, FLORA, "heldout", "TP8_vs_FLoRA.heldout", 7.85e-5),
        (RA, FLORA, "heldout", "RA_vs_FLoRA.heldout", 6.23e-3),
        (RA, TP8, "heldout", "RA_vs_TP8.heldout", 2.62e-2),
        (FFA, RA, "heldout", "FFA_vs_RA.heldout", 3.81e-4),
        (FEDIT, FLORA, "heldout", "FedIT_vs_FLoRA.heldout", 0.425),
    ]:
        add(f"ptest.{label}", "Appendix C Table 9", p, abs(p) * 0.02 + 1e-9,
            lambda x=a, y=b, m=metric: d.paired_p(x, y, m))

    # T-N1 p-value column: paired tests against FLoRA on final loss.
    for exp, label, pv in [(FEDIT, "FedIT", 0.5879), (TP8, "Two-Phase K=8", 4.182e-5),
                           (RA, "ReverseAdaptive", 1.087e-5), (FFA, "FFA-LoRA", 1.71e-4)]:
        add(f"tn1.pcol.{label}", "T-N1 col 6 (p vs FLoRA)", pv, abs(pv) * 0.02 + 1e-9,
            lambda e=exp: d.paired_p(e, FLORA, "loss"))

    # Metric-agreement claim in G5.5: identical ranking on the four distinct
    # operating points, with the FLoRA/FedIT pair the sole exception. Guards
    # against reinstating the "agree exactly" overclaim.
    def ranking_agrees_except_flora_fedit():
        by_loss = sorted([e for e, _ in FRONTIER], key=lambda e: d.final_loss(e)[0])
        by_held = sorted([e for e, _ in FRONTIER], key=lambda e: d.heldout(e)[0])
        collapse = lambda L: [("FLoRA_FedIT" if x in (FLORA, FEDIT) else x) for x in L]
        return 1 if collapse(by_loss) == collapse(by_held) and by_loss != by_held else 0

    add("g5.metric_ranking_agrees_on_distinct_points", "Sec 4.2 G5.5 paragraph 1", 1, 0,
        ranking_agrees_except_flora_fedit)
    add("g5.RA_minus_FFA_MB", "Sec 4.2 G5.5 (550 MB)", 550.0, 1e-6,
        lambda: d.comm(RA) - d.comm(FFA))
    add("g5.max_seed_sd_loss", "Sec 4.2 G5.5 (at most 0.0016)", 0.0016, 5e-5,
        lambda: max(d.final_loss(e)[1] for e, _ in FRONTIER))
    add("g5.max_seed_sd_heldout", "Sec 4.2 G5.5 (at most 0.0008)", 0.0008, 5e-5,
        lambda: max(d.heldout(e)[1] for e, _ in FRONTIER))
    add("g5.spread_ratio", "Sec 4.2 G5.7 (roughly 25x)", 25.02, 0.3,
        lambda: (d.final_loss(RA)[0] - d.final_loss(FLORA)[0]) / d.final_loss(RA)[1])

    # Bonferroni disclosure: the one cell that must fail at 5 comparisons.
    add("ptest.bonferroni_failing_cell", "Appendix C text (disclosed failure)", 1, 0,
        lambda: 1 if d.paired_p(RA, TP8, "heldout") > 0.05 / 5 else 0)

    # -- T-N2: Dolly replication -------------------------------------------
    add("dolly.comm.FLoRA", "T-N2", 2578.125, 1e-6, lambda: d.comm(D_FLORA))
    add("dolly.comm.TP8", "T-N2", 1863.125, 1e-6, lambda: d.comm(D_TP8))
    add("dolly.comm.RA", "T-N2", 1533.125, 1e-6, lambda: d.comm(D_RA))
    add("dolly.gap.RA_vs_FLoRA", "T-N2, Abstract (0.0061)", 0.006119, 5e-6,
        lambda: d.heldout(D_RA)[0] - d.heldout(D_FLORA)[0])
    add("dolly.gap_vs_alpaca_delta", "Sec 4.x cross-dataset stability", 0.000187, 5e-6,
        lambda: abs((d.heldout(RA)[0] - d.heldout(FLORA)[0])
                    - (d.heldout(D_RA)[0] - d.heldout(D_FLORA)[0])))


    # -- LLaMA-3.2-3B -------------------------------------------------------


    # -- T-N2 detail (G6) --------------------------------------------------
    for exp, label, loss, held in [
        (D_FLORA, "FLoRA", 1.6544, -0.5330),
        (D_TP8, "Two-Phase K=8", 1.6643, -0.5287),
        (D_RA, "ReverseAdaptive", 1.6686, -0.5268),
    ]:
        add(f"tn2.loss.{label}", "T-N2 col 3", loss, 5e-5, lambda e=exp: d.final_loss(e)[0])
        add(f"tn2.heldout.{label}", "T-N2 col 4", held, 5e-5, lambda e=exp: d.heldout(e)[0])

    add("tn2.gap.TP8.dolly", "T-N2 col 5", 0.004265, 5e-6,
        lambda: d.heldout(D_TP8)[0] - d.heldout(D_FLORA)[0])
    add("tn2.gap.TP8.alpaca", "T-N2 col 6", 0.003399, 5e-6,
        lambda: d.heldout(TP8)[0] - d.heldout(FLORA)[0])
    add("tn2.relshift.RA", "Sec 4.3 G6.1 (3.0%)", 3.0, 0.05,
        lambda: abs((d.heldout(D_RA)[0] - d.heldout(D_FLORA)[0])
                    - (d.heldout(RA)[0] - d.heldout(FLORA)[0]))
                / (d.heldout(RA)[0] - d.heldout(FLORA)[0]) * 100)
    add("tn2.relshift.TP8", "Sec 4.3 G6.1 (25.5%)", 25.5, 0.05,
        lambda: abs((d.heldout(D_TP8)[0] - d.heldout(D_FLORA)[0])
                    - (d.heldout(TP8)[0] - d.heldout(FLORA)[0]))
                / (d.heldout(TP8)[0] - d.heldout(FLORA)[0]) * 100)
    add("tn2.dolly_noniid_comm", "Sec 4.3 G6.1 (non-IID Dolly = 1533.13)", 1533.125, 1e-6,
        lambda: d.comm("exp_dolly_reverse_adaptive_noniid_alpha05"))
    for a, b, m, label, pv in [
        (D_TP8, D_FLORA, "loss", "dolly.TP8_vs_FLoRA.loss", 7.822e-5),
        (D_RA, D_FLORA, "loss", "dolly.RA_vs_FLoRA.loss", 2.525e-5),
        (D_TP8, D_FLORA, "heldout", "dolly.TP8_vs_FLoRA.heldout", 2.406e-6),
        (D_RA, D_FLORA, "heldout", "dolly.RA_vs_FLoRA.heldout", 7.597e-3),
    ]:
        add(f"ptest.{label}", "Appendix C Table 9 (Dolly)", pv, abs(pv) * 0.02 + 1e-9,
            lambda x=a, y=b, mm=m: d.paired_p(x, y, mm))

    add("l3.comm.FLoRA", "Sec 4.4, Table 6", 2625.0, 1e-6, lambda: d.l3_comm("flora")[0])
    add("l3.comm.FLoRA.n", "Sec 4.4 (3 seeds, smoke tests excluded)", 3, 0,
        lambda: d.l3_comm("flora")[2])
    add("l3.comm.RA", "Sec 4.4, Table 6", 1837.5, 1e-6, lambda: d.l3_comm("reverse_adaptive")[0])
    add("l3.comm.TP8", "Sec 4.4, Table 6", 1942.5, 1e-6, lambda: d.l3_comm("two_phase")[0])
    add("l3.savings.mean", "Abstract, Sec 4.4 (30.0%)", 30.0, 0.05,
        lambda: d.l3_savings_pct()[0])
    add("l3.savings.sd", "Abstract, Sec 4.4 (+/- 4.0)", 4.0, 0.05,
        lambda: d.l3_savings_pct()[1])
    add("l3.ptest.RA_vs_TP8", "Abstract, Sec 4.4, Sec 5.2 (p=0.997)", 0.997, 0.002,
        lambda: d.l3_paired_p("reverse_adaptive", "two_phase"))

    # -- G7: Section 5.2 claims --------------------------------------------
    add("g7.RA_worse_than_TP8_at_1B", "Sec 5.2 title (must NOT say 'outperforms')", 1, 0,
        lambda: 1 if d.final_loss(RA)[0] > d.final_loss(TP8)[0] else 0)
    add("g7.effect_size_ratio", "Sec 5.2 (roughly 350x)", 346.0, 5.0,
        lambda: (d.final_loss(RA)[0] - d.final_loss(TP8)[0])
                / abs(st.mean(d.l3("reverse_adaptive").final_loss)
                      - st.mean(d.l3("two_phase").final_loss)))
    add("g7.seg1_cost_per_pp", "Sec 5.2 (1.56e-4 per point)", 1.5558e-4, 5e-8,
        lambda: (d.heldout(RA)[0] - d.heldout(FLORA)[0])
                / ((d.comm(FLORA) - d.comm(RA)) / d.comm(FLORA) * 100))
    add("g7.seg2_cost_per_pp", "Sec 5.2 (8.54e-4 per point)", 8.5406e-4, 5e-8,
        lambda: (d.heldout(FFA)[0] - d.heldout(RA)[0])
                / ((d.comm(RA) - d.comm(FFA)) / d.comm(FLORA) * 100))
    add("g7.equiv_fixed_K", "Sec 5.2 (behaviorally equivalent to K=5)", 5, 0,
        lambda: int(sorted(set(d.runs.loc[d.runs.exp_name == RA, "switch_round"]))[0]) - 1)

    # -- Byte payloads (Sec 3.4, G4.2) -------------------------------------
    add("bytes.full.TinyLlama", "Sec 3.4, G4.2 baseline paragraph",
        FULL_PAYLOAD_BYTES_TINYLLAMA, 0, lambda: d.broadcast_bytes(FLORA))
    add("bytes.bonly.TinyLlama", "G4.2 baseline paragraph",
        BONLY_PAYLOAD_BYTES_TINYLLAMA, 0, lambda: d.broadcast_bytes(FFA))
    add("bytes.bonly_is_36pct", "Sec 3.1, Sec 3.4 (exactly 36%)", 0.36, 1e-12,
        lambda: BONLY_PAYLOAD_BYTES_TINYLLAMA / FULL_PAYLOAD_BYTES_TINYLLAMA)
    add("bytes.full_matches_param_count", "Sec 3.4 (2,252,800 params x 4 bytes)",
        FULL_PAYLOAD_BYTES_TINYLLAMA, 0, lambda: 102_400 * 22 * 4)

    # -- G8: transition cost and the FFA-LoRA overcharge --------------------
    # Naive total ignoring the transition upgrade: u = d = s-1.
    def naive(s_):
        return TL_STEP * (2 * s_ - 2) + TL_ALL_BONLY

    for exp, label, s_ in [(FFA, "FFA-LoRA", 1), (RA, "ReverseAdaptive", 6),
                           (TP8, "Two-Phase K=8", 9)]:
        add(f"g8.transition_penalty.{label}", "Sec 5.3 (fixed 55.0 MB)", 55.0, 1e-6,
            lambda e=exp, x=s_: d.comm(e) - naive(x))
    add("g8.symmetric_would_cost", "Sec 5.3 (110.0 MB)", 110.0, 1e-6,
        lambda: TL_STEP * 2)
    add("g8.ffa_faithful_floor", "Sec 5.3 (928.1 MB floor)", TL_ALL_BONLY, 1e-6,
        lambda: d.comm(FFA) - 55.0)
    add("g8.ffa_faithful_savings", "Sec 5.3 (64.00%)", 64.0, 5e-3,
        lambda: (d.comm(FLORA) - TL_ALL_BONLY) / d.comm(FLORA) * 100)
    add("g8.faithful_segment_pp", "Sec 5.3 (23.5 pp)", 23.47, 0.02,
        lambda: (d.comm(RA) - TL_ALL_BONLY) / d.comm(FLORA) * 100)

    def faithful_ratio():
        s1 = (d.comm(FLORA) - d.comm(RA)) / d.comm(FLORA) * 100
        q1 = d.heldout(RA)[0] - d.heldout(FLORA)[0]
        s2 = (d.comm(RA) - TL_ALL_BONLY) / d.comm(FLORA) * 100
        q2 = d.heldout(FFA)[0] - d.heldout(RA)[0]
        return (q2 / s2) / (q1 / s1)

    add("g8.faithful_marginal_ratio", "Sec 5.3 (5.5 becomes 5.0)", 4.99, 0.02, faithful_ratio)
    # Both readings must round to "roughly five times", which is what the
    # abstract and conclusion say.
    add("g8.roughly_five_both_ways", "Abstract, Conclusion (roughly five times)", 1, 0,
        lambda: 1 if (4.5 <= faithful_ratio() <= 5.5 and 4.5 <= marginal_ratio() <= 5.6) else 0)

    # -- Step function (Sec 5.3, new) --------------------------------------
    # Every switching method is one point on total = 55(2s-1) + 928.125.
    for exp, label, s in [(FFA, "FFA-LoRA", 1), (RA, "ReverseAdaptive", 6),
                          (TP8, "Two-Phase K=8", 9)]:
        add(f"stepfn.{label}", "Sec 5.3 step-function claim", tl_total(s), 1e-6,
            lambda e=exp: d.comm(e))
    add("stepfn.FLoRA_is_all_full_endpoint", "Sec 5.3", tl_no_switch(), 1e-6,
        lambda: d.comm(FLORA))
    add("stepfn.FLoRA_not_on_switch_lattice", "Sec 5.3 (must not claim s=16)", 1, 0,
        lambda: 1 if abs(tl_total(N_ROUNDS + 1) - d.comm(FLORA)) > 1e-6 else 0)
    add("stepfn.granularity_MB", "Sec 5.3, C14 (110 MB per round)", 110.0, 1e-9,
        lambda: tl_total(7) - tl_total(6))
    add("stepfn.k10_equals_s11", "Sec 5.3 (K=10 total = 2083.125)", 2083.125, 1e-6,
        lambda: tl_total(11))
    add("stepfn.l3_RA", "Sec 4.4 (3B on same step function, s=8)", l3_total(8), 1e-6,
        lambda: d.l3_comm("reverse_adaptive")[0])

    # -- G8b: benchmark insensitivity (Sec 5.5) ----------------------------
    BENCH_BASE = {"arc_easy_acc": 0.274, "boolq_acc": 0.626, "hellaswag_acc": 0.448}

    def tiny_bench(col):
        b = d.final[(d.final.scale == "tinyllama_1b") & d.final[col].notna()]
        return sorted(b[col].unique())

    for col, lo, hi in [("arc_easy_acc", -2.8, -2.2), ("boolq_acc", -7.4, -6.4),
                        ("hellaswag_acc", -2.8, -2.8)]:
        v = None
        add(f"g8b.regress_lo.{col}", "Sec 5.5 regression range", lo, 0.05,
            lambda c=col: (min(tiny_bench(c)) - BENCH_BASE[c]) * 100)
        add(f"g8b.regress_hi.{col}", "Sec 5.5 regression range", hi, 0.05,
            lambda c=col: (max(tiny_bench(c)) - BENCH_BASE[c]) * 100)
        add(f"g8b.spread.{col}", "Sec 5.5 cross-method spread",
            {"arc_easy_acc": 0.006, "boolq_acc": 0.010, "hellaswag_acc": 0.0}[col], 5e-4,
            lambda c=col: max(tiny_bench(c)) - min(tiny_bench(c)))

    # C22: the clustering figure must exclude MMLU, which the text excludes as
    # chance-level. Including it gives 1.4 pp; the three informative benchmarks
    # give 1.0 pp.
    add("g8b.cluster_pp_informative_only", "Table 3 caption, Sec 4.3, Conclusion (1.0 pp)",
        1.0, 0.05,
        lambda: max((max(tiny_bench(c)) - min(tiny_bench(c))) * 100 for c in BENCH_BASE))
    add("g8b.cluster_pp_would_be_with_mmlu", "C22 guard (must NOT be quoted)", 1.4, 0.05,
        lambda: (max(tiny_bench("mmlu_acc")) - min(tiny_bench("mmlu_acc"))) * 100)
    add("g8b.tiny_bench_is_single_seed", "Sec 5.5, Table 3 caption (C21)", 1, 0,
        lambda: 1 if set(d.final[(d.final.scale == "tinyllama_1b")
                                 & d.final.boolq_acc.notna()].seed) == {42} else 0)
    add("g8b.tuned_heldout_loss_lo", "Sec 5.5 (1.3349)", 1.3349, 5e-5,
        lambda: d.hold.loc[d.hold.dataset.astype(str).str.contains("alpaca", case=False,
                                                                   na=False), "tuned_loss"].min())
    add("g8b.tuned_heldout_loss_hi", "Sec 5.5 (1.3608)", 1.3608, 5e-5,
        lambda: d.hold.loc[d.hold.dataset.astype(str).str.contains("alpaca", case=False,
                                                                   na=False), "tuned_loss"].max())

    # -- G10 / C23: commit provenance --------------------------------------
    # T-N1 spans two run batches: FLoRA, Two-Phase K=8 and ReverseAdaptive from
    # phase1_cuda_rerun (commit 7b957752); FedIT and FFA-LoRA from
    # neuro_part12_qvonly (commit cae4c328). The FLoRA/FedIT pair straddles that
    # boundary and is statistically indistinguishable, which bounds the
    # revision effect empirically.
    def batch_of(exp):
        v = d.runs.loc[d.runs.exp_name == exp, "run_path"].astype(str)
        for b in ("phase1_cuda_rerun", "neuro_part12_qvonly", "neuro_part1_part2_4090"):
            if v.str.contains(b).all():
                return b
        return "MIXED"

    add("c23.tn1_spans_two_batches", "Sec 5.3/Limitations commit disclosure", 2, 0,
        lambda: len({batch_of(e) for e, _ in FRONTIER}))
    add("c23.flora_fedit_straddle", "Limitations (revision-effect control)", 1, 0,
        lambda: 1 if batch_of(FLORA) != batch_of(FEDIT) else 0)
    add("c23.revision_effect_bound", "Limitations (bounded at ~0.0006)", 0.000566, 5e-6,
        lambda: abs(d.final_loss(FEDIT)[0] - d.final_loss(FLORA)[0]))
    add("c23.tn2_single_batch", "T-N2 is single-revision", 1, 0,
        lambda: 1 if len({batch_of(e) for e in (D_FLORA, D_TP8, D_RA)}) == 1 else 0)

    # -- G11 / G13: byte accounting is revision-invariant by measurement ----
    # Three protocols were run on both sides of the revision boundary, since
    # the Dolly batch (cae4c328) reuses FLoRA, Two-Phase K=8 and
    # ReverseAdaptive. Identical totals prove invariance directly.
    for a, b, label in [(FLORA, D_FLORA, "FLoRA"), (TP8, D_TP8, "Two-Phase K=8"),
                        (RA, D_RA, "ReverseAdaptive")]:
        add(f"g13.byte_invariance.{label}", "Appendix D revision-invariance claim", 0.0, 1e-9,
            lambda x=a, y=b: d.comm(x) - d.comm(y))

    # -- G11: Appendix B corpora ------------------------------------------
    def a01():
        f = d.final
        return f[(f.scale == "tinyllama_1b") & (f.setting == "noniid_alpha01")
                 & (f.method == "reverse_adaptive")]

    add("g11.alpha01_seed_count", "Table B2 (five MPS seeds)", 5, 0, lambda: len(a01()))
    add("g11.alpha01_loss_min", "Table B2 spread", 1.259200, 5e-6,
        lambda: a01().final_loss.min())
    add("g11.alpha01_loss_max", "Table B2 spread", 1.348083, 5e-6,
        lambda: a01().final_loss.max())
    add("g11.alpha01_9clients", "Table B2 caption (9/10 of 1533.13)", 1379.8125, 1e-6,
        lambda: tl_total(6) * 9 / 10)
    add("g11.alpha01_8clients", "Table B2 caption (8/10 of 1533.13)", 1226.5, 1e-6,
        lambda: tl_total(6) * 8 / 10)
    add("g11.alpha01_comm_observed", "Table B2 (observed totals match client scaling)", 1, 0,
        lambda: 1 if set(round(x, 4) for x in a01().total_mb)
                     <= {1533.125, 1379.8125, 1226.5} else 0)

    # -- G9: switch-disabling behavior (Sec 5.4) ---------------------------
    # Per-round training loss of the canonical MPS FLoRA seed-42 run
    # (results/raw/exp_flora_iid/flora/seed_42/run_07_of_18/20260428_153342),
    # the run the threshold ablation in Table 5 is built on. Embedded here
    # because it is a published input to that table and is not in the CSVs.
    FLORA_S42 = [1.37814102, 1.30326323, 1.29772416, 1.29519047, 1.28920123,
                 1.28554564, 1.28219286, 1.27574476, 1.27433958, 1.27003106,
                 1.27117855, 1.27070879, 1.26548198, 1.26015272, 1.25943428]

    def _rho():
        return [(FLORA_S42[i - 1] - FLORA_S42[i]) / FLORA_S42[i - 1]
                for i in range(1, len(FLORA_S42))]

    def reconstruct_switch(tau, W=5):
        rho = _rho()
        for r in range(2, len(FLORA_S42) + 1):
            if r > W and rho[r - 2] < tau:
                return r
        return None

    # The published ablation must be predictable from the FLoRA trajectory
    # alone. This confirms ReverseAdaptive tracks FLoRA exactly pre-switch,
    # which Section 4.3 asserts but never quantifies.
    for tau, expected in [(0.005, 6), (0.010, 6), (0.002, 9), (0.001, 11)]:
        add(f"g9.ablation_reconstructs.tau{tau}", "Table 5, Sec 5.4", expected, 0,
            lambda t=tau: reconstruct_switch(t))

    # tau = 0 does NOT disable the switch: the loss rises once, at round 11.
    add("g9.tau_zero_still_fires", "Sec 5.4 (tau=0 fires at round 11)", 11, 0,
        lambda: reconstruct_switch(0.0))
    add("g9.most_negative_rho", "Sec 5.4 (approx -0.0009)", -0.000903, 5e-6,
        lambda: min(_rho()[5:]))
    add("g9.negative_tau_disables", "Sec 5.4 (tau below min rho disables)", 1, 0,
        lambda: 1 if reconstruct_switch(-0.001) is None else 0)

    # -- Switch-round claims -----------------------------------------------
    add("switch.TinyLlama_all_round6", "Sec 4.5, G7 saturation disclosure", 1, 0,
        lambda: 1 if set(
            d.runs.loc[d.runs.exp_name.isin([RA, D_RA]), "switch_round"].astype(str)
        ) == {"6"} else 0)

    # The manuscript reports SAMPLE standard deviation. The 3B figures are
    # dispositive: savings sd is exactly 4.0000 and communication sd exactly
    # 105.00 under sample sd, versus 3.266 and 85.73 under population sd.
    # This assertion prevents the convention drifting between tables.
    add("convention.sd_is_sample", "All tables, all captions", 4.0, 1e-9,
        lambda: d.l3_savings_pct()[1])
    add("convention.sd_not_population", "All tables (guard)", 1, 0,
        lambda: 1 if abs(st.pstdev([(2625 - m) / 2625 * 100
                                    for m in d.l3("reverse_adaptive").total_mb]) - 4.0) > 0.1 else 0)

    return A


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="analysis")
    ap.add_argument("--filter", default=None,
                    help="substring match on assertion id or manuscript location")
    ap.add_argument("--show-computed", action="store_true")
    args = ap.parse_args()

    d = Data(args.analysis_dir)
    try:
        A = build(d)
    except Exception as e:  # noqa: BLE001
        print(f"FATAL while building assertions: {type(e).__name__}: {e}")
        return 2

    if args.filter:
        f = args.filter.lower()
        A = [a for a in A if f in a[0].lower() or f in a[1].lower()]

    npass = nfail = nerr = 0
    failures = []
    print(f"{'status':6}  {'id':38} {'expected':>14} {'computed':>14}  location")
    print("-" * 110)
    for aid, loc, expected, tol, fn in A:
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            nerr += 1
            failures.append((aid, loc, f"{type(e).__name__}: {e}"))
            print(f"{'ERROR':6}  {aid:38} {expected!s:>14} {'--':>14}  {loc}")
            continue
        ok = (got == expected) if tol == 0 else (abs(got - expected) <= tol)
        if ok:
            npass += 1
            if args.show_computed:
                print(f"{'ok':6}  {aid:38} {expected:>14} {got:>14.8g}  {loc}")
            else:
                print(f"{'ok':6}  {aid:38} {expected!s:>14} {'':>14}  {loc}")
        else:
            nfail += 1
            failures.append((aid, loc, f"expected {expected}, computed {got!r}"))
            print(f"{'FAIL':6}  {aid:38} {expected!s:>14} {got:>14.8g}  {loc}")

    print("-" * 110)
    print(f"{npass} passed, {nfail} failed, {nerr} errored, {len(A)} total")
    if failures:
        print("\nEdit these manuscript locations:")
        for aid, loc, msg in failures:
            print(f"  [{aid}] {loc}\n      {msg}")
    return 0 if (nfail == 0 and nerr == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for federated checkpoint save / resume (Neurocomputing Phase 0 C.8)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

from src.federation.aggregators.ffa_lora import FFALoRAAggregator
from src.federation.aggregators.reverse_adaptive import ReverseAdaptiveAggregator
from src.federation.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    resolve_checkpoint_path,
    restore_rng_state,
    run_dir_from_checkpoint,
)
from src.federation.server import FederatedServer


def _tiny_lora_state(seed: int = 0) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(
            2, 4, generator=g
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(
            4, 2, generator=g
        ),
    }


class _StubClient:
    """Deterministic client: mixes broadcast state with a fixed local delta."""

    def __init__(self, client_id: int, base: Dict[str, torch.Tensor]):
        self.client_id = client_id
        self.base = {k: v.clone() for k, v in base.items()}

    def train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        freeze_a: bool = False,
        b_only_upload: bool = False,
        **kwargs,
    ) -> Dict:
        # Draw a few randoms so RNG restore matters.
        _ = torch.rand(3)
        state = {
            k: (global_state[k].float() if global_state is not None else self.base[k].float())
            + 0.01 * (self.client_id + 1)
            for k in self.base
        }
        if b_only_upload:
            state = {k: v for k, v in state.items() if "lora_B" in k or "lora_b" in k}
        return {
            "state_dict": state,
            "loss": float(1.0 / (1 + self.client_id) + torch.rand(1).item() * 0.01),
            "num_samples": 10,
            "training_time": 0.0,
        }


def test_ffa_state_dict_roundtrip():
    agg = FFALoRAAggregator()
    states = [_tiny_lora_state(i) for i in range(3)]
    out = agg.aggregate(states)
    sd = agg.state_dict()
    agg2 = FFALoRAAggregator()
    agg2.load_state_dict(sd)
    assert agg2.initialized
    for k, v in agg.frozen_a.items():
        assert torch.allclose(agg2.frozen_a[k], v)
    # Next aggregate with B-only-ish states still works
    out2 = agg2.aggregate(states)
    assert set(out2.keys()) == set(out.keys())


def test_reverse_adaptive_state_dict_roundtrip():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01,
        warmup_rounds=2,
        transition_rounds=0,
        max_rank=2,
    )
    for r in range(1, 5):
        states = [_tiny_lora_state(r * 10 + i) for i in range(2)]
        loss = 1.5 - 0.01 * r  # tiny improvements → switch after warmup
        agg.aggregate(states, weights=[1.0, 1.0], current_loss=loss, round_num=r)
    sd = agg.state_dict()
    agg2 = ReverseAdaptiveAggregator(
        switch_threshold=0.01,
        warmup_rounds=2,
        transition_rounds=0,
        max_rank=2,
    )
    agg2.load_state_dict(sd)
    assert agg2.round_count == agg.round_count
    assert agg2.current_mode == agg.current_mode
    assert agg2.loss_history == agg.loss_history
    assert agg2.ffa_lora.initialized == agg.ffa_lora.initialized


def test_rng_capture_restore():
    torch.manual_seed(0)
    _ = torch.rand(5)
    snap = capture_rng_state()
    a = torch.rand(5).clone()
    restore_rng_state(snap)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_server_checkpoint_resume_continues_rounds(tmp_path: Path):
    torch.manual_seed(42)
    np.random.seed(42)
    base = _tiny_lora_state(0)
    out1 = tmp_path / "run_a"
    server = FederatedServer(
        aggregation_method="reverse_adaptive",
        num_rounds=4,
        eval_every=100,
        output_dir=str(out1),
        lora_r=2,
        reverse_adaptive={
            "switch_threshold": 0.01,
            "warmup_rounds": 5,
            "transition_rounds": 0,
        },
        save_every=2,
        keep_last_n=3,
        seed=42,
    )
    server.set_clients([_StubClient(i, base) for i in range(3)])
    results_full = server.train()
    assert len(results_full["metrics"]) == 4
    ckpt = resolve_checkpoint_path(out1)
    assert ckpt.name == "latest.pt"
    loaded = load_checkpoint(ckpt)
    assert loaded["completed_rounds"] == 4  # last save at round 4? save_every=2 → 2,4
    # Actually last completed is 4; save at 2 and 4. latest should be 4.
    assert loaded["completed_rounds"] == 4

    # Interrupted-style: stop after round 2, resume
    out2 = tmp_path / "run_b"
    server2 = FederatedServer(
        aggregation_method="reverse_adaptive",
        num_rounds=4,
        eval_every=100,
        output_dir=str(out2),
        lora_r=2,
        reverse_adaptive={
            "switch_threshold": 0.01,
            "warmup_rounds": 5,
            "transition_rounds": 0,
        },
        save_every=2,
        keep_last_n=3,
        seed=42,
    )
    server2.set_clients([_StubClient(i, base) for i in range(3)])
    # Manually mimic first 2 rounds then checkpoint via a short run
    short = FederatedServer(
        aggregation_method="reverse_adaptive",
        num_rounds=2,
        eval_every=100,
        output_dir=str(out2),
        lora_r=2,
        reverse_adaptive={
            "switch_threshold": 0.01,
            "warmup_rounds": 5,
            "transition_rounds": 0,
        },
        save_every=2,
        keep_last_n=3,
        seed=42,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    short.set_clients([_StubClient(i, base) for i in range(3)])
    short.train()
    mid = resolve_checkpoint_path(out2)
    assert load_checkpoint(mid)["completed_rounds"] == 2

    torch.manual_seed(999)  # scramble; load_checkpoint should restore
    np.random.seed(999)
    server2.load_checkpoint(str(mid))
    assert server2._resume_start_round == 2
    resumed = server2.train()
    assert len(resumed["metrics"]) == 4
    assert [m["round"] for m in resumed["metrics"]] == [1, 2, 3, 4]
    # Round 3 loss within atol=0.01 of uninterrupted reference
    ref3 = results_full["metrics"][2]["avg_loss"]
    got3 = resumed["metrics"][2]["avg_loss"]
    assert abs(ref3 - got3) < 0.01


def test_run_dir_from_checkpoint(tmp_path: Path):
    run = tmp_path / "exp" / "seed_42" / "ts"
    ckpt = run / "checkpoints" / "latest.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"completed_rounds": 1}, ckpt)
    assert run_dir_from_checkpoint(ckpt) == run.resolve()

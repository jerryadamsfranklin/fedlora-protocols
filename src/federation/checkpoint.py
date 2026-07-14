"""
Federated training checkpoints (Neurocomputing Phase 0 / C.8).

A checkpoint is written after completed communication rounds and stores:
  - LoRA / global adapter weights
  - aggregator state (mode, frozen A, loss history, …)
  - federated round index and cumulative communication counters
  - RNG state (python / numpy / torch[/cuda])
  - optimizer_state: empty with metadata — clients construct a fresh AdamW
    each local round, so there is no durable cross-round optimizer

Resume continues at completed_rounds + 1 with restored RNG so the next-round
loss can be compared to an uninterrupted reference (atol=0.01).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

CHECKPOINT_VERSION = 1
CHECKPOINT_DIRNAME = "checkpoints"
LATEST_NAME = "latest.pt"


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            pass


def resolve_checkpoint_path(path: Union[str, Path]) -> Path:
    """
    Accept a .pt file, a checkpoints/ dir, or a run output dir containing
    checkpoints/latest.pt.
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        latest = p / LATEST_NAME
        if latest.is_file():
            return latest
        nested = p / CHECKPOINT_DIRNAME / LATEST_NAME
        if nested.is_file():
            return nested
        # Prefer highest round_XXXX.pt if latest missing
        round_files = sorted(p.glob("round_*.pt"))
        if not round_files and (p / CHECKPOINT_DIRNAME).is_dir():
            round_files = sorted((p / CHECKPOINT_DIRNAME).glob("round_*.pt"))
        if round_files:
            return round_files[-1]
    raise FileNotFoundError(f"No checkpoint found at {path}")


def run_dir_from_checkpoint(ckpt_path: Path) -> Path:
    """
    checkpoints/round_0003.pt -> run output dir (parent of checkpoints/).
    If the file lives directly in the run dir, return that dir.
    """
    ckpt_path = Path(ckpt_path).resolve()
    parent = ckpt_path.parent
    if parent.name == CHECKPOINT_DIRNAME:
        return parent.parent
    return parent


def save_checkpoint(path: Union[str, Path], payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    path = resolve_checkpoint_path(path)
    try:
        data = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        data = torch.load(path, map_location=map_location)
    if not isinstance(data, dict) or "completed_rounds" not in data:
        raise ValueError(f"Invalid checkpoint format: {path}")
    return data


def prune_old_checkpoints(ckpt_dir: Union[str, Path], keep_last_n: int) -> None:
    if keep_last_n is None or keep_last_n < 1:
        return
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return
    rounds = sorted(ckpt_dir.glob("round_*.pt"))
    for old in rounds[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


def write_round_checkpoint(
    output_dir: Union[str, Path],
    *,
    completed_rounds: int,
    keep_last_n: int = 3,
    **payload: Any,
) -> Path:
    """Write round_XXXX.pt and update latest.pt under output_dir/checkpoints/."""
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / CHECKPOINT_DIRNAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "version": CHECKPOINT_VERSION,
        "completed_rounds": int(completed_rounds),
        **payload,
    }
    round_path = ckpt_dir / f"round_{int(completed_rounds):04d}.pt"
    save_checkpoint(round_path, body)
    latest_path = ckpt_dir / LATEST_NAME
    save_checkpoint(latest_path, body)
    prune_old_checkpoints(ckpt_dir, keep_last_n)
    return latest_path

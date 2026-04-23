"""
Stability Detection Mixin for FedLoRA-Adaptive v2.

Detects training instability (loss spikes) and can trigger reversion to FFA-LoRA.
"""

from __future__ import annotations

from typing import List, Optional


class StabilityDetectorMixin:
    """
    Mixin that provides stability detection functionality.

    Monitors loss trajectory and detects:
    - Single-round spikes (loss increased > threshold)
    - Sustained instability (loss increased for N consecutive rounds)
    """

    def init_stability_detector(
        self,
        spike_threshold: float = 1.1,  # 10% increase = spike
        sustained_rounds: int = 2,  # N rounds of increase = unstable
    ) -> None:
        self.spike_threshold = float(spike_threshold)
        self.sustained_rounds = int(sustained_rounds)
        self.loss_history: List[float] = []
        self.revert_count = 0
        self.last_stable_round = 0

    def record_loss(self, loss: float, round_num: Optional[int] = None) -> None:
        _ = round_num
        self.loss_history.append(float(loss))

    def detect_instability(self, current_loss: float) -> bool:
        if len(self.loss_history) < 1:
            return False

        current_loss = float(current_loss)
        previous_loss = self.loss_history[-1]

        # Single-round spike
        if current_loss > previous_loss * self.spike_threshold:
            return True

        # Sustained increase for N rounds (based on history up to previous round).
        # Expected usage: call detect_instability(current_loss) BEFORE recording it.
        n = self.sustained_rounds
        if n >= 2 and len(self.loss_history) >= n:
            recent = self.loss_history[-n:]
            all_increasing = all(
                recent[i] > recent[i - 1] for i in range(1, len(recent))
            )
            if all_increasing:
                return True

        return False

    def detect_instability_detailed(self, current_loss: float) -> dict:
        result = {"unstable": False, "reason": None}
        if len(self.loss_history) < 1:
            return result

        current_loss = float(current_loss)
        previous_loss = self.loss_history[-1]

        if current_loss > previous_loss * self.spike_threshold:
            increase_pct = (
                (current_loss - previous_loss) / previous_loss * 100
                if previous_loss != 0
                else float("inf")
            )
            result["unstable"] = True
            result["reason"] = f"Loss spike: +{increase_pct:.1f}%"
            return result

        n = self.sustained_rounds
        if n >= 2 and len(self.loss_history) >= n:
            recent = self.loss_history[-n:]
            all_increasing = all(
                recent[i] > recent[i - 1] for i in range(1, len(recent))
            )
            if all_increasing:
                result["unstable"] = True
                result["reason"] = f"Sustained increase for {n} rounds"
                return result

        return result

    def record_revert(self, round_num: Optional[int] = None) -> None:
        _ = round_num
        self.revert_count += 1

    def get_stability_stats(self) -> dict:
        return {
            "revert_count": self.revert_count,
            "loss_history_length": len(self.loss_history),
            "last_stable_round": self.last_stable_round,
            "spike_threshold": self.spike_threshold,
            "sustained_rounds": self.sustained_rounds,
        }


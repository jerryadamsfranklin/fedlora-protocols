"""
Federated Learning Server

Coordinates training:
1. Initialize global model
2. For each round:
   a. Send global state to clients
   b. Clients train locally
   c. Aggregate client updates
   d. Evaluate if needed
3. Save results

CURSOR AI: Implement this file as specified.
"""

import json
import os
import time
import inspect
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .aggregators.fedit import FedITAggregator
from .aggregators.ffa_lora import FFALoRAAggregator
from .aggregators.flora import FLoRAAggregator
from .aggregators.flexlora import FlexLoRAAggregator
from .aggregators.reverse_adaptive import ReverseAdaptiveAggregator
from .aggregators.two_phase import TwoPhaseAggregator


class FederatedServer:
    """
    Server for federated learning simulation.
    """

    AGGREGATORS = {
        "fedit": FedITAggregator,
        "ffa_lora": FFALoRAAggregator,
        "flora": FLoRAAggregator,
        "flexlora": FlexLoRAAggregator,
        "two_phase": TwoPhaseAggregator,
        "reverse_adaptive": ReverseAdaptiveAggregator,
    }

    def __init__(
        self,
        aggregation_method: str = "fedit",
        num_rounds: int = 30,
        eval_every: int = 5,
        output_dir: str = "results",
        lora_r: int = 16,
        switch_threshold: float = 0.01,
        warmup_rounds: int = 3,
        transition_rounds: int = 3,
        stability_threshold: float = 1.1,
        two_phase: Optional[Dict[str, Any]] = None,
        reverse_adaptive: Optional[Dict[str, Any]] = None,
    ):
        self.num_rounds = num_rounds
        self.eval_every = eval_every
        self.output_dir = output_dir

        # Initialize aggregator
        if aggregation_method not in self.AGGREGATORS:
            raise ValueError(f"Unknown method: {aggregation_method}")
        self.aggregation_method = aggregation_method
        # FLoRA/FlexLoRA SVD outputs must match PEFT adapter shapes (rank r).
        agg_cls = self.AGGREGATORS[aggregation_method]
        if aggregation_method == "flora":
            self.aggregator = agg_cls(max_rank=lora_r)
        elif aggregation_method == "flexlora":
            self.aggregator = agg_cls(global_rank=lora_r)
        elif aggregation_method == "two_phase":
            tp = two_phase or {}
            self.aggregator = TwoPhaseAggregator(
                phase_boundary=int(tp.get("phase_boundary", 8)),
                max_rank=lora_r,
            )
        elif aggregation_method == "reverse_adaptive":
            ra = reverse_adaptive or {}
            self.aggregator = ReverseAdaptiveAggregator(
                switch_threshold=float(ra.get("switch_threshold", 0.01)),
                warmup_rounds=int(ra.get("warmup_rounds", 5)),
                transition_rounds=int(ra.get("transition_rounds", 2)),
                stability_threshold=float(ra.get("stability_threshold", 1.1)),
                max_rank=lora_r,
            )
        else:
            self.aggregator = agg_cls()

        self.global_state = None
        self.clients: List = []
        self.metrics_history = []

        os.makedirs(output_dir, exist_ok=True)

    def set_clients(self, clients: List) -> None:
        """Set the list of clients."""
        self.clients = clients

    @staticmethod
    def _aggregate_layer_metrics(
        all_layer_metrics: List[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        if not all_layer_metrics:
            return None
        all_layers = set()
        for m in all_layer_metrics:
            all_layers.update(m.keys())
        aggregated: Dict[str, float] = {}
        for layer in all_layers:
            vals = [m[layer] for m in all_layer_metrics if layer in m]
            if vals:
                aggregated[layer] = float(sum(vals) / len(vals))
        return aggregated

    def train(self, eval_fn=None) -> Dict:
        """
        Run federated training.

        Args:
            eval_fn: Optional function(state_dict) -> metrics dict

        Returns:
            Dict with final results
        """
        print(f"\n{'='*60}")
        print(f"Federated Training: {self.aggregator.name}")
        print(f"Clients: {len(self.clients)}, Rounds: {self.num_rounds}")
        print(f"{'='*60}\n")

        total_upload_bytes = 0
        total_download_bytes = 0

        for round_num in range(self.num_rounds):
            round_start = time.time()
            print(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

            # Collect client updates
            client_states = []
            client_weights = []
            losses = []
            all_layer_metrics: List[Dict[str, float]] = []

            if hasattr(self.aggregator, "get_freeze_a"):
                freeze_a = bool(self.aggregator.get_freeze_a())
            else:
                freeze_a = self.aggregation_method == "ffa_lora"

            # Communication optimization: upload B-only when the aggregator says
            # it's safe (i.e., after frozen A has been initialized server-side).
            b_only_upload = False
            if freeze_a and hasattr(self.aggregator, "should_upload_b_only"):
                try:
                    b_only_upload = bool(self.aggregator.should_upload_b_only())
                except Exception:
                    b_only_upload = False

            freeze_ratios = None
            if hasattr(self.aggregator, "get_layer_freeze_ratios"):
                try:
                    freeze_ratios = self.aggregator.get_layer_freeze_ratios(
                        round_num + 1
                    )
                except Exception:
                    freeze_ratios = None

            for client in tqdm(self.clients, desc="Training"):
                if freeze_ratios is not None:
                    result = client.train(
                        self.global_state,
                        freeze_ratios=freeze_ratios,
                        collect_layer_metrics=True,
                    )
                else:
                    result = client.train(
                        self.global_state,
                        freeze_a=freeze_a,
                        b_only_upload=b_only_upload,
                    )
                client_states.append(result["state_dict"])
                client_weights.append(result["num_samples"])
                losses.append(result["loss"])
                if "layer_metrics" in result:
                    all_layer_metrics.append(result["layer_metrics"])

                # Track communication (upload: client -> server)
                total_upload_bytes += sum(
                    p.numel() * p.element_size()
                    for p in result["state_dict"].values()
                )

            # Reconstruction: merge B-only uploads with frozen A so downstream
            # aggregators still receive full (A+B) states.
            if b_only_upload and hasattr(self.aggregator, "get_frozen_a"):
                try:
                    frozen_a = self.aggregator.get_frozen_a()
                except Exception:
                    frozen_a = {}
                if frozen_a:
                    for state in client_states:
                        for a_key, a_tensor in frozen_a.items():
                            if a_key not in state:
                                state[a_key] = a_tensor.clone()

            # Aggregate (some aggregators accept extra signals like current_loss)
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            aggregated_layer_metrics = self._aggregate_layer_metrics(all_layer_metrics)
            sig = inspect.signature(self.aggregator.aggregate)
            call_kw: Dict[str, Any] = {"client_states": client_states}
            if "weights" in sig.parameters:
                call_kw["weights"] = client_weights
            if "round_num" in sig.parameters:
                call_kw["round_num"] = round_num + 1
            if "current_loss" in sig.parameters:
                call_kw["current_loss"] = avg_loss
            if "layer_metrics" in sig.parameters:
                call_kw["layer_metrics"] = aggregated_layer_metrics
            self.global_state = self.aggregator.aggregate(**call_kw)

            # Add download communication (server -> each client)
            bytes_per_client = sum(
                p.numel() * p.element_size() for p in self.global_state.values()
            )
            total_download_bytes += bytes_per_client * len(self.clients)

            round_time = time.time() - round_start

            # Log
            total_communication_mb = (total_upload_bytes + total_download_bytes) / (
                1024 * 1024
            )
            metrics = {
                "round": round_num + 1,
                "avg_loss": avg_loss,
                "round_time": round_time,
                "communication_mb": total_communication_mb,
                "upload_mb": total_upload_bytes / (1024 * 1024),
                "download_mb": total_download_bytes / (1024 * 1024),
            }
            if hasattr(self.aggregator, "get_mode"):
                metrics["agg_mode"] = self.aggregator.get_mode()
            if hasattr(self.aggregator, "get_switch_round"):
                metrics["switch_round"] = self.aggregator.get_switch_round()
            if hasattr(self.aggregator, "get_stats"):
                metrics["aggregator_stats"] = self.aggregator.get_stats()

            # Evaluate
            if eval_fn and (round_num + 1) % self.eval_every == 0:
                eval_metrics = eval_fn(self.global_state)
                metrics.update(eval_metrics)
                print(f"  Eval: {eval_metrics}")

            self.metrics_history.append(metrics)
            print(f"  Loss: {avg_loss:.4f}, Time: {round_time:.1f}s")

        # Save results
        self._save_results()

        return {
            "final_state": self.global_state,
            "metrics": self.metrics_history,
            "total_communication_mb": (total_upload_bytes + total_download_bytes)
            / (1024 * 1024),
            "total_upload_mb": total_upload_bytes / (1024 * 1024),
            "total_download_mb": total_download_bytes / (1024 * 1024),
        }

    def _save_results(self) -> None:
        """Save results to JSON."""
        path = os.path.join(self.output_dir, "results.json")
        # Make metrics JSON-serializable (e.g. numpy floats)
        serializable = []
        for m in self.metrics_history:
            row = {}
            for k, v in m.items():
                if hasattr(v, "item"):
                    row[k] = v.item()
                else:
                    row[k] = v
            serializable.append(row)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {path}")

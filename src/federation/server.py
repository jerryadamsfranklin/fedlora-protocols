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
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .aggregators.fedit import FedITAggregator
from .aggregators.ffa_lora import FFALoRAAggregator
from .aggregators.fedlora_adaptive import FedLoRAAdaptiveAggregator
from .aggregators.fedlora_adaptive_v2 import FedLoRAAdaptiveV2Aggregator
from .aggregators.flora import FLoRAAggregator
from .aggregators.flexlora import FlexLoRAAggregator


class FederatedServer:
    """
    Server for federated learning simulation.
    """

    AGGREGATORS = {
        "fedit": FedITAggregator,
        "ffa_lora": FFALoRAAggregator,
        "flora": FLoRAAggregator,
        "flexlora": FlexLoRAAggregator,
        "fedlora_adaptive": FedLoRAAdaptiveAggregator,
        "fedlora_adaptive_v2": FedLoRAAdaptiveV2Aggregator,
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
        fixed_switch_round: Optional[int] = None,
        transition_rounds: int = 3,
        stability_threshold: float = 1.1,
        per_layer_enabled: bool = True,
        layer_names: Optional[List[str]] = None,
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
        elif aggregation_method == "fedlora_adaptive":
            self.aggregator = agg_cls(
                max_rank=lora_r,
                switch_threshold=switch_threshold,
                warmup_rounds=warmup_rounds,
                fixed_switch_round=fixed_switch_round,
            )
        elif aggregation_method == "fedlora_adaptive_v2":
            self.aggregator = agg_cls(
                layer_names=layer_names or ["q_proj", "k_proj", "v_proj", "o_proj"],
                max_rank=lora_r,
                switch_threshold=switch_threshold,
                warmup_rounds=warmup_rounds,
                transition_rounds=transition_rounds,
                stability_threshold=stability_threshold,
                per_layer_enabled=per_layer_enabled,
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

        total_communication = 0

        for round_num in range(self.num_rounds):
            round_start = time.time()
            print(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

            # Collect client updates
            client_states = []
            client_weights = []
            losses = []
            all_layer_metrics: List[Dict[str, float]] = []

            if self.aggregation_method == "fedlora_adaptive" and hasattr(
                self.aggregator, "get_mode"
            ):
                freeze_a = self.aggregator.get_mode() == "FFA-LoRA"
            else:
                freeze_a = self.aggregation_method == "ffa_lora"

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
                    result = client.train(self.global_state, freeze_a=freeze_a)
                client_states.append(result["state_dict"])
                client_weights.append(result["num_samples"])
                losses.append(result["loss"])
                if "layer_metrics" in result:
                    all_layer_metrics.append(result["layer_metrics"])

                # Track communication (upload: client -> server)
                total_communication += sum(
                    p.numel() * 2 for p in result["state_dict"].values()
                )

            # Aggregate (FedLoRA-Adaptive needs current_loss for switching)
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            aggregated_layer_metrics = self._aggregate_layer_metrics(all_layer_metrics)
            try:
                import inspect

                sig = inspect.signature(self.aggregator.aggregate)
                if "current_loss" in sig.parameters:
                    kwargs = {
                        "client_states": client_states,
                        "weights": client_weights,
                        "current_loss": avg_loss,
                    }
                    if "layer_metrics" in sig.parameters:
                        kwargs["layer_metrics"] = aggregated_layer_metrics
                    if "round_num" in sig.parameters:
                        kwargs["round_num"] = round_num + 1
                    self.global_state = self.aggregator.aggregate(**kwargs)
                else:
                    self.global_state = self.aggregator.aggregate(
                        client_states,
                        weights=client_weights,
                    )
            except Exception:
                self.global_state = self.aggregator.aggregate(
                    client_states,
                    weights=client_weights,
                )

            # Add download communication (server -> each client)
            total_communication += (
                sum(p.numel() * 2 for p in self.global_state.values())
                * len(self.clients)
            )

            round_time = time.time() - round_start

            # Log
            metrics = {
                "round": round_num + 1,
                "avg_loss": avg_loss,
                "round_time": round_time,
                "communication_mb": total_communication / (1024 * 1024),
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
            "total_communication_mb": total_communication / (1024 * 1024),
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

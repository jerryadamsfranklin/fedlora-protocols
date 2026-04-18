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
    }

    def __init__(
        self,
        aggregation_method: str = "fedit",
        num_rounds: int = 30,
        eval_every: int = 5,
        output_dir: str = "results",
    ):
        self.num_rounds = num_rounds
        self.eval_every = eval_every
        self.output_dir = output_dir

        # Initialize aggregator
        if aggregation_method not in self.AGGREGATORS:
            raise ValueError(f"Unknown method: {aggregation_method}")
        self.aggregation_method = aggregation_method
        self.aggregator = self.AGGREGATORS[aggregation_method]()

        self.global_state = None
        self.clients: List = []
        self.metrics_history = []

        os.makedirs(output_dir, exist_ok=True)

    def set_clients(self, clients: List) -> None:
        """Set the list of clients."""
        self.clients = clients

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

            freeze_a = self.aggregation_method == "ffa_lora"
            for client in tqdm(self.clients, desc="Training"):
                result = client.train(self.global_state, freeze_a=freeze_a)
                client_states.append(result["state_dict"])
                client_weights.append(result["num_samples"])
                losses.append(result["loss"])

                # Track communication (upload: client -> server)
                total_communication += sum(
                    p.numel() * 2 for p in result["state_dict"].values()
                )

            # Aggregate
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
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # Log
            metrics = {
                "round": round_num + 1,
                "avg_loss": avg_loss,
                "round_time": round_time,
                "communication_mb": total_communication / (1024 * 1024),
            }

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

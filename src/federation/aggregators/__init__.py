"""
Federated LoRA aggregation methods.

Note: keep this package lightweight to avoid importing heavyweight deps (e.g. torch)
at module import time. Individual aggregators should be imported from their modules:

- `src.federation.aggregators.fedit`
- `src.federation.aggregators.ffa_lora`
- `src.federation.aggregators.flora`
- `src.federation.aggregators.flexlora`
- `src.federation.aggregators.fedlora_adaptive`
- `src.federation.aggregators.fedlora_adaptive_v2`
- `src.federation.aggregators.two_phase`
- `src.federation.aggregators.reverse_adaptive`
- `src.federation.aggregators.budget_adaptive`
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Type

__all__ = [
    "get_aggregator_cls",
    "AGGREGATOR_IMPORTS",
]


# Lazy registry (module path, class name). Keeps import-time lightweight.
AGGREGATOR_IMPORTS: Dict[str, tuple[str, str]] = {
    "fedit": ("src.federation.aggregators.fedit", "FedITAggregator"),
    "ffa_lora": ("src.federation.aggregators.ffa_lora", "FFALoRAAggregator"),
    "flora": ("src.federation.aggregators.flora", "FLoRAAggregator"),
    "flexlora": ("src.federation.aggregators.flexlora", "FlexLoRAAggregator"),
    "fedlora_adaptive": (
        "src.federation.aggregators.fedlora_adaptive",
        "FedLoRAAdaptiveAggregator",
    ),
    "fedlora_adaptive_v2": (
        "src.federation.aggregators.fedlora_adaptive_v2",
        "FedLoRAAdaptiveV2Aggregator",
    ),
    "two_phase": ("src.federation.aggregators.two_phase", "TwoPhaseAggregator"),
    "reverse_adaptive": (
        "src.federation.aggregators.reverse_adaptive",
        "ReverseAdaptiveAggregator",
    ),
    "budget_adaptive": (
        "src.federation.aggregators.budget_adaptive",
        "BudgetAdaptiveAggregator",
    ),
    "curriculum_rank": (
        "src.federation.aggregators.curriculum_rank",
        "CurriculumRankAggregator",
    ),
}


def get_aggregator_cls(method: str) -> Type:
    """Resolve an aggregator class from the lazy registry."""
    if method not in AGGREGATOR_IMPORTS:
        raise KeyError(f"Unknown aggregator method: {method}")
    mod_path, cls_name = AGGREGATOR_IMPORTS[method]
    mod = import_module(mod_path)
    return getattr(mod, cls_name)

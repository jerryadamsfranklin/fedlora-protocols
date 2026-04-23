"""
Federated LoRA aggregation methods.

Note: keep this package lightweight to avoid importing heavyweight deps (e.g. torch)
at module import time. Individual aggregators should be imported from their modules:

- `src.federation.aggregators.fedit`
- `src.federation.aggregators.ffa_lora`
- `src.federation.aggregators.flora`
- `src.federation.aggregators.flexlora`
- `src.federation.aggregators.fedlora_adaptive`
"""

__all__ = []

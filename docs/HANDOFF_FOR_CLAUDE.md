# Federated LoRA experiments — handoff package for Claude

This document bundles **what the project is**, the **timing / compute problem**, **observed results**, and **full source code** from the repository so another assistant can continue without repo access.

---

## 1. Purpose of this repo

Independent research comparing **four federated LoRA aggregation strategies** under identical setups:

| Method | Role |
|--------|------|
| **FedIT** (`fedit`) | FedAvg-style average of LoRA A/B |
| **FFA-LoRA** (`ffa_lora`) | Freeze A; train & aggregate B |
| **FLoRA** (`flora`) | Stack LoRAs, compress with SVD |
| **FlexLoRA** (`flexlora`) | Aggregate in ΔW space via weighted Σ(BA), SVD back to rank-r |

Hardware target: **Mac Apple Silicon**, **MPS**, primary model **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`**.

Full methodological spec (RQ1–RQ5, configs, experiment matrix) lives in **`Federated_LoRA_Complete_Guide.md`** (~2000 lines) at repo root — not pasted here.

---

## 2. Timing / compute problem (critical)

### 2.1 What the guide assumed

The guide estimated roughly **30–40 hours wall time** for **all** experiments combined (IID + non-IID + rank sweep + client scaling).

### 2.2 What actually happened on hardware

Runs are **sequential**: each **round** trains **10 clients** one after another; each client runs **`local_epochs: 2`** over its shard with **`batch_size: 4`**, **`gradient_accumulation_steps: 4`**, **`max_seq_length: 512`**, **`max_samples: 5000`** partitioned IID → **500 samples/client**.

**Measured per-round wall time** (FedIT, EXP1, float32/MPS fix): **`round_time` ≈ 5,100–5,800 seconds** (~**85–97 minutes/round**) in saved `results.json`.

Implication:

- One full run (**30 rounds**, one method):  
  \( 30 × \sim 87\text{ min} \approx \mathbf{43\text{–}45\text{ hours}} \) wall time.

Full planned suite from the YAML design is **28 separate runs**:

- EXP1: 4 methods  
- EXP2: 4 methods  
- EXP3: 4 methods  
- EXP4: 2 methods × 5 ranks = 10 runs *(not automated in CLI yet — needs loops / separate configs)*  
- EXP5: 2 methods × 3 client counts = 6 runs *(same)*  

Naïve extrapolation worst case:

- \( 28 × 43\text{ h} \approx \mathbf{1200+\ hours} \) (~**50 days** continuous) — **does not match** the guide’s 30–40h estimate.

### 2.3 Additional factors that increased time / risk

1. **`float16` on MPS** initially produced **`Loss: nan`** → runs were invalid until **`model.torch_dtype: float32`** and training fixes (train LoRA only, grad clip, skip non-finite loss).

2. **Float32** is slower/heavier than FP16 but **numerically stable** on MPS.

3. **`analyze_results.py`**, **`generate_figures.py`**, **`run_all_experiments.sh`** are **still empty stubs** — no automated batch analysis yet.

### 2.4 What to do next (research-integrity friendly)

Without “cheating”, any reduction must be **documented as the protocol** for the paper:

- Fewer **`num_rounds`** (e.g. 30 → 15–20 after pilot convergence curves),
- **`local_epochs: 1`** instead of 2,
- Lower **`max_samples`**,
- Or stratified workflow: cheap **pilots** then one **canonical** protocol for tables.

Fair comparison still requires **same** protocol across all four methods.

---

## 3. Bundled results (EXP1 IID, FedIT — completed run)

**Output directory:** `results/exp1_iid_fedit_20260415_233512/`  
**Artifact:** `results.json` — array of `{ round, avg_loss, round_time, communication_mb }`.

Embedded in full below (30 rounds).

### Summary statistics (from JSON)

| Metric | Approx |
|--------|--------|
| Initial `avg_loss` (round 1) | ~1.285 |
| Final `avg_loss` (round 30) | ~0.958 |
| `round_time` range | ~5,086 – 5,763 s |
| Mean `round_time` | ~5,257 s (**~87.6 min**) |
| Total serial training time | ~**43.8 h** |
| Final `communication_mb` (tracked as cumulative in server) | ~2,578 MB |

### Full `results.json`

```json
[
  {"round": 1, "avg_loss": 1.2846466885328294, "round_time": 5241.866749048233, "communication_mb": 85.9375},
  {"round": 2, "avg_loss": 1.2342403504967687, "round_time": 5195.88481092453, "communication_mb": 171.875},
  {"round": 3, "avg_loss": 1.2212780636906622, "round_time": 5253.98387503624, "communication_mb": 257.8125},
  {"round": 4, "avg_loss": 1.2089016321182249, "round_time": 5762.484140872955, "communication_mb": 343.75},
  {"round": 5, "avg_loss": 1.2008668996572494, "round_time": 5721.169562101364, "communication_mb": 429.6875},
  {"round": 6, "avg_loss": 1.191065773272514, "round_time": 5607.005274057388, "communication_mb": 515.625},
  {"round": 7, "avg_loss": 1.1798492659807205, "round_time": 5464.957503080368, "communication_mb": 601.5625},
  {"round": 8, "avg_loss": 1.172053145623207, "round_time": 5086.925758123398, "communication_mb": 687.5},
  {"round": 9, "avg_loss": 1.1600880971074106, "round_time": 5276.976507902145, "communication_mb": 773.4375},
  {"round": 10, "avg_loss": 1.1498124931693077, "round_time": 5302.551802873611, "communication_mb": 859.375},
  {"round": 11, "avg_loss": 1.1405192454338073, "round_time": 5333.456167221069, "communication_mb": 945.3125},
  {"round": 12, "avg_loss": 1.1304872028827666, "round_time": 5264.165818929672, "communication_mb": 1031.25},
  {"round": 13, "avg_loss": 1.1188965020656587, "round_time": 5190.449155092239, "communication_mb": 1117.1875},
  {"round": 14, "avg_loss": 1.1102218315005303, "round_time": 5183.941689014435, "communication_mb": 1203.125},
  {"round": 15, "avg_loss": 1.100553767335415, "round_time": 5147.769744873047, "communication_mb": 1289.0625},
  {"round": 16, "avg_loss": 1.0910603718280791, "round_time": 5142.120673894882, "communication_mb": 1375.0},
  {"round": 17, "avg_loss": 1.0809877311944962, "round_time": 5153.854692935944, "communication_mb": 1460.9375},
  {"round": 18, "avg_loss": 1.0718600660324096, "round_time": 5203.084299087524, "communication_mb": 1546.875},
  {"round": 19, "avg_loss": 1.061078145134449, "round_time": 5137.639366865158, "communication_mb": 1632.8125},
  {"round": 20, "avg_loss": 1.0522727319002152, "round_time": 5308.33433508873, "communication_mb": 1718.75},
  {"round": 21, "avg_loss": 1.042242704176903, "round_time": 5295.92342877388, "communication_mb": 1804.6875},
  {"round": 22, "avg_loss": 1.0335175481796264, "round_time": 5296.059080839157, "communication_mb": 1890.625},
  {"round": 23, "avg_loss": 1.026442392528057, "round_time": 5113.164171934128, "communication_mb": 1976.5625},
  {"round": 24, "avg_loss": 1.0134306301236153, "round_time": 5253.893164873123, "communication_mb": 2062.5},
  {"round": 25, "avg_loss": 1.0042524502396581, "round_time": 5250.019938707352, "communication_mb": 2148.4375},
  {"round": 26, "avg_loss": 0.997079615688324, "round_time": 5121.656897068024, "communication_mb": 2234.375},
  {"round": 27, "avg_loss": 0.986410210621357, "round_time": 5269.021767139435, "communication_mb": 2320.3125},
  {"round": 28, "avg_loss": 0.9760779478549958, "round_time": 5309.026969909668, "communication_mb": 2406.25},
  {"round": 29, "avg_loss": 0.9676963133096695, "round_time": 5210.567045927048, "communication_mb": 2492.1875},
  {"round": 30, "avg_loss": 0.9583988858342171, "round_time": 5293.59858417511, "communication_mb": 2578.125}
]
```

---

## 4. Run command

```bash
conda activate fedlora
cd /path/to/federated-lora-experiments
python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit
```

---

## 5. Repository file tree (implementation-relevant)

```
config/
  base_config.yaml
  exp1_iid.yaml … exp5_client_scaling.yaml
scripts/
  run_experiment.py
  analyze_results.py          # empty stub
  generate_figures.py           # empty stub
  run_all_experiments.sh        # empty stub
src/
  models/lora_model.py
  data/data_partitioner.py
  federation/client.py
  federation/server.py
  federation/aggregators/{base,fedit,ffa_lora,flora,flexlora}.py
  evaluation/{metrics,statistical_tests}.py   # empty stubs
  utils/{config,logging_utils,memory_utils}.py # empty stubs
requirements.txt
.gitignore
```

---

## 6. Full source code

### `requirements.txt`

```
torch==2.2.0
torchvision
torchaudio
transformers==4.40.0
accelerate==0.29.0
peft==0.10.0
datasets==2.18.0
evaluate==0.4.1
numpy==1.26.0
scipy==1.12.0
scikit-learn==1.4.0
pandas==2.2.0
matplotlib==3.8.0
seaborn==0.13.0
pyyaml==6.0.1
tqdm==4.66.0
wandb==0.16.0
huggingface_hub
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.tox/
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# macOS
.DS_Store

# Logs / experiment artifacts
wandb/
*.log

# Local caches
.cache/

# Data/results outputs (keep .gitkeep placeholders)
results/*
!results/.gitkeep
figures/*
!figures/.gitkeep
```

### `config/base_config.yaml`

```yaml
# =============================================================================
# BASE CONFIGURATION
# All experiment configs inherit from this and can override specific values
# =============================================================================

# Model Configuration
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Primary model for experiments
  # Alternative: "meta-llama/Llama-3.2-3B" for larger experiments
  torch_dtype: "float32"
  device: "mps"  # Apple Silicon GPU

# LoRA Configuration
lora:
  r: 16                          # Rank of LoRA matrices
  lora_alpha: 32                 # Scaling factor
  lora_dropout: 0.1              # Dropout probability
  target_modules:                # Attention layers to adapt
    - "q_proj"
    - "v_proj"
  bias: "none"

# Local Training Configuration (per client per round)
training:
  local_epochs: 2                # How many epochs each client trains
  batch_size: 4                  # Samples per batch
  gradient_accumulation_steps: 4 # Effective batch = 4 * 4 = 16
  learning_rate: 0.0002          # 2e-4
  weight_decay: 0.01
  max_seq_length: 512
  warmup_ratio: 0.03

# Federated Learning Configuration
federated:
  num_rounds: 30                 # Total communication rounds
  num_clients: 10                # Number of simulated clients
  clients_per_round: 10          # Full participation
  aggregation_method: "fedit"    # Will be overridden per run

# Evaluation Configuration
evaluation:
  eval_every: 5                  # Evaluate every N rounds
  eval_batch_size: 8
  metrics:
    - "loss"
    - "perplexity"

# Checkpointing
checkpointing:
  save_every: 10                 # Save checkpoint every N rounds
  keep_last_n: 3                 # Keep last N checkpoints

# Reproducibility
seed: 42
```

### `config/exp1_iid.yaml`

```yaml
# =============================================================================
# EXPERIMENT 1: IID BASELINE
# =============================================================================
# 
# RESEARCH QUESTION: RQ1
# How do the 4 aggregation methods compare when client data is IID?
#
# HYPOTHESIS:
# All methods should perform similarly under IID conditions, as there is
# no data heterogeneity to expose differences in robustness.
#
# EXPECTED OUTCOME:
# - Similar final loss for all methods
# - FLoRA/FlexLoRA may converge slightly faster (less aggregation noise)
# - Clear differences in communication cost
#
# =============================================================================

experiment:
  name: "exp1_iid"
  description: "IID baseline comparison of all 4 aggregation methods"
  research_question: "RQ1"

# Inherit base config
_inherit: "base_config.yaml"

# Data configuration
data:
  dataset_name: "tatsu-lab/alpaca"
  dataset_split: "train"
  max_samples: 5000              # Use subset for tractable experiments
  partition_method: "iid"        # Random uniform split across clients
  eval_samples: 500              # Held-out samples for evaluation

# Methods to run (all 4)
methods: ["fedit", "ffa_lora", "flora", "flexlora"]

# Expected runtime: ~6-8 hours total (4 methods × ~1.5-2 hours each)
```

### `config/exp2_noniid_label.yaml`

```yaml
# =============================================================================
# EXPERIMENT 2: NON-IID LABEL SKEW
# =============================================================================
#
# RESEARCH QUESTION: RQ2
# How does non-IID data distribution affect each method?
#
# WHAT IS LABEL SKEW:
# Different clients have different distributions of labels/categories.
# Example: Hospital A sees mostly cancer patients, Hospital B sees mostly
# cardiac patients. They have the same task but different data distributions.
#
# HOW WE SIMULATE IT:
# Use Dirichlet distribution with alpha=0.5 to assign labels to clients.
# Lower alpha = more skew (each client gets fewer label types).
#
# HYPOTHESIS (H1):
# FFA-LoRA will outperform FedIT under non-IID conditions because
# freezing the A matrix reduces sensitivity to heterogeneous gradients.
#
# EXPECTED OUTCOME:
# - FedIT shows largest performance degradation vs IID
# - FFA-LoRA, FLoRA, FlexLoRA more robust
# - Clear ranking emerges: FlexLoRA ≥ FLoRA ≥ FFA-LoRA > FedIT
#
# =============================================================================

experiment:
  name: "exp2_noniid_label"
  description: "Non-IID with label distribution skew (Dirichlet alpha=0.5)"
  research_question: "RQ2"

_inherit: "base_config.yaml"

data:
  dataset_name: "commonsense_qa"  # Multiple-choice QA (has clear labels)
  dataset_split: "train"
  partition_method: "label_skew"
  dirichlet_alpha: 0.5           # Concentration parameter (lower = more skew)
  label_column: "answerKey"      # Column containing labels (A, B, C, D, E)
  eval_split: "validation"
  eval_samples: 500

# Override for shorter sequences in QA
training:
  max_seq_length: 256

methods: ["fedit", "ffa_lora", "flora", "flexlora"]

# Expected runtime: ~6-8 hours total
```

### `config/exp3_noniid_quantity.yaml`

```yaml
# =============================================================================
# EXPERIMENT 3: NON-IID QUANTITY SKEW
# =============================================================================
#
# RESEARCH QUESTION: RQ2 (continued)
# How do methods handle clients with vastly different amounts of data?
#
# WHAT IS QUANTITY SKEW:
# Different clients have different amounts of data.
# Example: Large hospital has 10,000 records, small clinic has 100 records.
#
# HOW WE SIMULATE IT:
# Use Dirichlet distribution to assign data quantities.
# Some clients get 10x more data than others.
#
# HYPOTHESIS:
# Methods that weight by sample count (FedAvg weighting) will be more
# robust. FlexLoRA's dynamic rank allocation helps under-resourced clients.
#
# EXPECTED OUTCOME:
# - Higher variance in results across runs
# - FedAvg weighting mitigates some effects
# - FlexLoRA should perform well (designed for heterogeneous resources)
#
# =============================================================================

experiment:
  name: "exp3_noniid_quantity"
  description: "Non-IID with quantity skew (imbalanced data amounts)"
  research_question: "RQ2"

_inherit: "base_config.yaml"

data:
  dataset_name: "tatsu-lab/alpaca"
  dataset_split: "train"
  max_samples: 5000
  partition_method: "quantity_skew"
  quantity_alpha: 0.3            # Lower = more imbalanced
  min_samples_per_client: 50     # Ensure no client is empty
  eval_samples: 500

methods: ["fedit", "ffa_lora", "flora", "flexlora"]

# Expected runtime: ~6-8 hours total
```

### `config/exp4_rank_sensitivity.yaml`

```yaml
# =============================================================================
# EXPERIMENT 4: RANK SENSITIVITY ANALYSIS
# =============================================================================
#
# RESEARCH QUESTION: RQ4
# How does LoRA rank (r) affect performance and communication cost?
#
# WHAT IS LORA RANK:
# r determines the size of A (r × hidden_dim) and B (hidden_dim × r).
# Higher r = more parameters = more expressive but more expensive.
#
# RANKS TO TEST: 4, 8, 16, 32, 64
#
# HYPOTHESIS:
# - Performance improves with rank up to a point (diminishing returns ~r=32)
# - Communication cost scales linearly with rank
# - FFA-LoRA may be more sensitive to rank (only B is trained)
#
# EXPECTED OUTCOME:
# - Curves showing performance vs rank for each method
# - Optimal rank identification
# - Pareto frontier of quality vs communication
#
# =============================================================================

experiment:
  name: "exp4_rank_sensitivity"
  description: "Analyze impact of LoRA rank on performance"
  research_question: "RQ4"

_inherit: "base_config.yaml"

data:
  dataset_name: "tatsu-lab/alpaca"
  dataset_split: "train"
  max_samples: 3000              # Smaller for faster iteration
  partition_method: "iid"
  eval_samples: 300

# Ranks to test
lora_ranks: [4, 8, 16, 32, 64]

# Only test 2 most relevant methods
methods: ["fedit", "ffa_lora"]

# Expected runtime: ~8-10 hours (2 methods × 5 ranks)
```

### `config/exp5_client_scaling.yaml`

```yaml
# =============================================================================
# EXPERIMENT 5: CLIENT SCALING
# =============================================================================
#
# RESEARCH QUESTION: RQ5
# How do methods perform as the number of clients increases?
#
# WHY THIS MATTERS:
# Real federations may have 10, 50, or 100+ participants.
# More clients = less data per client = potentially harder problem.
# Some methods (FLoRA) have communication that grows with clients.
#
# CLIENT COUNTS TO TEST: 5, 10, 20
#
# HYPOTHESIS:
# - All methods degrade somewhat with more clients
# - FLoRA communication grows (stacked matrices get bigger)
# - FedIT may be most affected (more heterogeneous gradients to average)
#
# =============================================================================

experiment:
  name: "exp5_client_scaling"
  description: "Analyze impact of number of clients"
  research_question: "RQ5"

_inherit: "base_config.yaml"

data:
  dataset_name: "tatsu-lab/alpaca"
  dataset_split: "train"
  max_samples: 5000
  partition_method: "iid"
  eval_samples: 500

# Client counts to test
client_counts: [5, 10, 20]

# Test 2 most relevant methods
methods: ["fedit", "flora"]

# Expected runtime: ~4-6 hours (2 methods × 3 client counts)
```

### `scripts/run_experiment.py`

```python
"""
Run a single federated LoRA experiment.

Usage:
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit

CURSOR AI: Implement this script as specified.
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_partitioner import DataPartitioner
from src.federation.client import FederatedClient
from src.federation.server import FederatedServer
from src.models.lora_model import FederatedLoRAModel


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base (override wins)."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config, resolving _inherit to merge with base config."""
    with open(path) as f:
        config = yaml.safe_load(f)

    if not config:
        return {}

    inherit = config.pop("_inherit", None)
    if inherit:
        config_dir = os.path.dirname(os.path.abspath(path))
        base_path = os.path.join(config_dir, inherit)
        with open(base_path) as bf:
            base = yaml.safe_load(bf)
        base = {k: v for k, v in base.items() if k != "_inherit"}
        config = _deep_merge(base, config)

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--method", default=None, help="Override aggregation method"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config (with base merge if _inherit present)
    config = load_config(args.config)

    # Override method if specified; else from config or methods[0]
    method = args.method
    if method is None:
        method = config.get("federated", {}).get(
            "aggregation_method",
            (config.get("methods") or ["fedit"])[0],
        )

    # Set seed
    torch.manual_seed(args.seed)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment", {}).get("name", "exp")
    output_dir = os.path.join(
        "results", f"{exp_name}_{method}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")

    # Device: MPS on Mac, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\n[1/4] Loading model...")
    model_cfg = config.get("model", {})
    torch_dtype = model_cfg.get("torch_dtype", "float32")
    model = FederatedLoRAModel(
        model_name=model_cfg.get(
            "name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        lora_r=config.get("lora", {}).get("r", 16),
        lora_alpha=config.get("lora", {}).get("lora_alpha", 32),
        device=device,
        torch_dtype=torch_dtype,
    )
    model.load_model()

    # Load data
    print("\n[2/4] Loading data...")
    data_cfg = config.get("data", {})
    dataset = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg.get("dataset_split", "train"),
    )
    if "max_samples" in data_cfg:
        dataset = dataset.select(
            range(min(data_cfg["max_samples"], len(dataset)))
        )

    # Partition
    num_clients = config.get("federated", {}).get("num_clients", 10)
    partitioner = DataPartitioner(dataset, num_clients=num_clients, seed=args.seed)
    partition_method = data_cfg.get("partition_method", "iid")

    if partition_method == "iid":
        client_datasets = partitioner.iid_partition()
    elif partition_method == "label_skew":
        client_datasets = partitioner.label_skew_partition(
            label_column=data_cfg.get("label_column", "label"),
            alpha=data_cfg.get("dirichlet_alpha", 0.5),
        )
    elif partition_method == "quantity_skew":
        client_datasets = partitioner.quantity_skew_partition(
            alpha=data_cfg.get("quantity_alpha", 0.5),
            min_samples=data_cfg.get("min_samples_per_client", 10),
        )
    else:
        client_datasets = partitioner.iid_partition()

    print(f"Partition stats: {partitioner.get_stats(client_datasets)}")

    # Create clients
    print("\n[3/4] Creating clients...")
    train_cfg = config.get("training", {})
    clients = []
    for i, ds in enumerate(client_datasets):
        if len(ds) == 0:
            continue
        client = FederatedClient(
            client_id=i,
            model=model,
            dataset=ds,
            batch_size=train_cfg.get("batch_size", 4),
            local_epochs=train_cfg.get("local_epochs", 2),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            max_seq_length=train_cfg.get("max_seq_length", 512),
            gradient_accumulation_steps=train_cfg.get(
                "gradient_accumulation_steps", 4
            ),
        )
        clients.append(client)

    if not clients:
        print("No clients with data. Exiting.")
        return

    # Create server
    fed_cfg = config.get("federated", {})
    eval_cfg = config.get("evaluation", {})
    server = FederatedServer(
        aggregation_method=method,
        num_rounds=fed_cfg.get("num_rounds", 30),
        eval_every=eval_cfg.get("eval_every", 5),
        output_dir=output_dir,
    )
    server.set_clients(clients)

    # Run training
    print("\n[4/4] Training...")
    results = server.train()

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Communication: {results['total_communication_mb']:.2f} MB")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
```

### `src/models/lora_model.py`

```python
"""
Federated LoRA Model Wrapper

This module wraps a HuggingFace model with LoRA adapters for use in
federated learning. It handles:
- Model loading with optional quantization
- LoRA configuration and application
- Extracting/loading only LoRA parameters for efficient communication
- Memory management for Apple Silicon

CURSOR AI: Implement this file with all methods as specified.
"""

import gc
from typing import Dict, List, Optional, Union

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class FederatedLoRAModel:
    """
    Wrapper for LoRA-adapted LLM in federated learning.

    Example usage:
        model = FederatedLoRAModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            lora_r=16,
            device="mps"
        )
        model.load_model()

        # After local training, get params to send to server
        state = model.get_lora_state_dict()

        # After receiving aggregated params from server
        model.set_lora_state_dict(aggregated_state)
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: str = "mps",
        torch_dtype: Union[str, torch.dtype] = "float32",
    ):
        """
        Initialize model wrapper (does not load model yet).

        Args:
            model_name: HuggingFace model ID
            lora_r: LoRA rank (higher = more expressive, more params)
            lora_alpha: LoRA scaling (effective scale = alpha/r)
            lora_dropout: Dropout on LoRA layers
            target_modules: Which layers to apply LoRA (default: q_proj, v_proj)
            device: "mps" for Mac, "cuda" for NVIDIA, "cpu" for CPU
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.device = device
        self.torch_dtype = torch_dtype

        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load base model and apply LoRA adapters."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        dtype = self._resolve_dtype(self.torch_dtype)
        print(f"torch_dtype: {dtype}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        # Training-friendly defaults
        if getattr(base_model.config, "use_cache", None) is not None:
            base_model.config.use_cache = False

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        # Apply LoRA
        self._model = get_peft_model(base_model, lora_config)
        self._model.print_trainable_parameters()

    @staticmethod
    def _resolve_dtype(torch_dtype: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if torch_dtype not in mapping:
            raise ValueError(
                f"Unsupported torch_dtype: {torch_dtype}. "
                f"Supported: {sorted(mapping.keys())}"
            )
        return mapping[torch_dtype]

    @property
    def model(self):
        """Get the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extract only LoRA parameters.

        Returns dict of {param_name: tensor} for all LoRA params.
        Tensors are on CPU for transmission.
        """
        state = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                state[name] = param.detach().cpu().clone()
        return state

    def set_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load LoRA parameters from state dict.

        Args:
            state_dict: Dict of LoRA parameters (from server aggregation)
        """
        current_state = self.model.state_dict()
        for name, param in state_dict.items():
            if name in current_state:
                current_state[name].copy_(param.to(self.device))

    def get_lora_param_count(self) -> int:
        """Count trainable LoRA parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_communication_cost_bytes(self) -> int:
        """Calculate bytes to transmit (one way, float16)."""
        return self.get_lora_param_count() * 2  # 2 bytes per float16

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
```

### `src/data/data_partitioner.py`

```python
"""
Data Partitioning for Federated Learning

This module splits datasets among clients for federated learning simulation.
Supports:
- IID: Each client gets random sample (ideal case)
- Label Skew: Clients have different label distributions (realistic)
- Quantity Skew: Clients have different amounts of data (realistic)

CURSOR AI: Implement this file with all methods as specified.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset


class DataPartitioner:
    """
    Partition dataset for federated learning simulation.

    Example:
        partitioner = DataPartitioner(dataset, num_clients=10, seed=42)

        # IID split
        client_datasets = partitioner.iid_partition()

        # Non-IID label skew
        client_datasets = partitioner.label_skew_partition(alpha=0.5)

        # Non-IID quantity skew
        client_datasets = partitioner.quantity_skew_partition(alpha=0.3)
    """

    def __init__(self, dataset: Dataset, num_clients: int, seed: int = 42):
        """
        Initialize partitioner.

        Args:
            dataset: HuggingFace Dataset to partition
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)

    def iid_partition(self) -> List[Dataset]:
        """
        IID partition: random uniform split.

        Each client gets len(dataset)/num_clients samples randomly.
        All clients have similar data distributions.
        """
        indices = np.random.permutation(len(self.dataset))
        splits = np.array_split(indices, self.num_clients)

        client_datasets = []
        for split_idx in splits:
            client_datasets.append(self.dataset.select(split_idx.tolist()))

        return client_datasets

    def label_skew_partition(
        self,
        label_column: str = "label",
        alpha: float = 0.5,
    ) -> List[Dataset]:
        """
        Non-IID partition with label distribution skew.

        Uses Dirichlet distribution to assign different proportions of
        each label to each client.

        Args:
            label_column: Column containing labels
            alpha: Dirichlet concentration (lower = more skew)
                   alpha=0.1: extreme skew (each client gets ~1 label)
                   alpha=0.5: moderate skew (recommended)
                   alpha=10: nearly IID
        """
        # Group indices by label
        label_to_indices = defaultdict(list)
        for idx, example in enumerate(self.dataset):
            label = example[label_column]
            label_to_indices[label].append(idx)

        # Initialize client index lists
        client_indices = [[] for _ in range(self.num_clients)]

        # Distribute each label's data according to Dirichlet
        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)

            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            counts = (proportions * len(indices)).astype(int)
            counts[-1] = len(indices) - counts[:-1].sum()  # Fix rounding

            # Assign to clients
            start = 0
            for client_id, count in enumerate(counts):
                client_indices[client_id].extend(indices[start : start + count])
                start += count

        # Create datasets
        client_datasets = []
        for indices in client_indices:
            np.random.shuffle(indices)
            client_datasets.append(self.dataset.select(indices))

        return client_datasets

    def quantity_skew_partition(
        self,
        alpha: float = 0.5,
        min_samples: int = 10,
    ) -> List[Dataset]:
        """
        Non-IID partition with quantity skew.

        Different clients get different amounts of data.

        Args:
            alpha: Dirichlet concentration (lower = more imbalanced)
            min_samples: Minimum samples per client
        """
        total = len(self.dataset)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * self.num_clients)

        # Ensure minimum samples
        base = np.array([min_samples] * self.num_clients)
        remaining = total - base.sum()
        if remaining <= 0:
            # Not enough data; give min_samples to each and split rest
            indices = np.random.permutation(total)
            splits = np.array_split(indices, self.num_clients)
            return [self.dataset.select(s.tolist()) for s in splits]
        extra = (proportions * remaining).astype(int)
        extra[-1] = remaining - extra[:-1].sum()

        sizes = base + extra

        # Shuffle and split
        indices = np.random.permutation(total)
        client_datasets = []
        start = 0
        for size in sizes:
            client_datasets.append(
                self.dataset.select(indices[start : start + size].tolist())
            )
            start += size

        return client_datasets

    def get_stats(self, client_datasets: List[Dataset]) -> Dict:
        """Get partition statistics."""
        sizes = [len(d) for d in client_datasets]
        return {
            "num_clients": len(client_datasets),
            "total_samples": sum(sizes),
            "min_samples": min(sizes),
            "max_samples": max(sizes),
            "mean_samples": float(np.mean(sizes)),
            "std_samples": float(np.std(sizes)),
        }
```

### `src/federation/client.py`

```python
"""
Federated Learning Client

Simulates a client in federated learning:
1. Receives global model state from server
2. Trains on local data
3. Returns updated model state

CURSOR AI: Implement this file with all methods as specified.
"""

import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling


class FederatedClient:
    """
    Simulates a federated learning client.

    Each client:
    - Has its own local dataset
    - Trains for local_epochs each round
    - Sends only LoRA parameters to server
    """

    def __init__(
        self,
        client_id: int,
        model,  # FederatedLoRAModel
        dataset,  # HuggingFace Dataset
        batch_size: int = 4,
        local_epochs: int = 2,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Initialize client.

        Args:
            client_id: Unique identifier for this client
            model: FederatedLoRAModel instance
            dataset: This client's local dataset
            batch_size: Training batch size
            local_epochs: Epochs to train each round
            learning_rate: Learning rate for optimizer
            max_seq_length: Max sequence length for tokenization
            gradient_accumulation_steps: Steps to accumulate gradients
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Prepare dataloader
        self._prepare_dataloader()

    def _prepare_dataloader(self):
        """Tokenize dataset and create dataloader."""
        def tokenize(examples):
            # Handle different dataset formats
            if "instruction" in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(
                        examples["instruction"], examples["output"]
                    )
                ]
            elif "question" in examples:
                texts = examples["question"]
            elif "text" in examples:
                texts = examples["text"]
            else:
                # Fallback: concatenate all string fields
                texts = [str(ex) for ex in examples[list(examples.keys())[0]]]

            return self.model.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )

        # Tokenize
        tokenized = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=self.dataset.column_names,
        )
        tokenized.set_format("torch")

        # Create dataloader
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )
        self.dataloader = DataLoader(
            tokenized,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

    def train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """
        Perform local training.

        Args:
            global_state: LoRA state from server (None for first round)

        Returns:
            Dict with:
            - state_dict: Updated LoRA parameters
            - loss: Average training loss
            - num_samples: Number of samples trained on
            - training_time: Time taken in seconds
        """
        # Load global state if provided
        if global_state is not None:
            self.model.set_lora_state_dict(global_state)

        # Setup optimizer
        trainable_params = [p for p in self.model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
        )

        # Training loop
        self.model.model.train()
        total_loss = 0.0
        num_steps = 0
        start_time = time.time()

        for epoch in range(self.local_epochs):
            for step, batch in enumerate(
                tqdm(
                    self.dataloader,
                    desc=f"Client {self.client_id} Epoch {epoch+1}",
                    leave=False,
                )
            ):
                # Move to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # Forward
                outputs = self.model.model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = loss / self.gradient_accumulation_steps

                # Backward
                loss.backward()

                # Update
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                num_steps += 1

        training_time = time.time() - start_time
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0

        # Get updated state
        updated_state = self.model.get_lora_state_dict()

        # Cleanup
        optimizer.zero_grad(set_to_none=True)
        self.model.clear_memory()

        return {
            "state_dict": updated_state,
            "loss": avg_loss,
            "num_samples": len(self.dataset),
            "training_time": training_time,
        }
```

### `src/federation/server.py`

```python
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

            for client in tqdm(self.clients, desc="Training"):
                result = client.train(self.global_state)
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
```

### `src/federation/aggregators/base.py`

```python
"""Base interface for federated LoRA aggregators."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch


class BaseAggregator(ABC):
    """Abstract base for LoRA aggregation methods."""

    name: str = "Base"

    @abstractmethod
    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client LoRA states into a global state."""
        pass

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client round-trip (upload + download)."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # float16 * 2 directions
```

### `src/federation/aggregators/fedit.py`

```python
"""
FedIT Aggregator: Baseline FedAvg on LoRA

Simply averages A and B matrices independently across clients.
This is the baseline method from Zhang et al., ICASSP 2024.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FedITAggregator:
    """
    FedIT: Standard FedAvg on LoRA parameters.

    Aggregation rule:
        global_A = Σ (w_k * A_k)
        global_B = Σ (w_k * B_k)

    where w_k is weight for client k.
    """

    def __init__(self):
        self.name = "FedIT"

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client states using weighted average.

        Args:
            client_states: List of LoRA state dicts from clients
            weights: Optional weights (default: uniform)

        Returns:
            Aggregated global state dict
        """
        if not client_states:
            raise ValueError("No client states to aggregate")

        # Default to uniform weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Aggregate each parameter
        aggregated = {}
        for name in client_states[0].keys():
            weighted_sum = sum(
                w * state[name].float()
                for w, state in zip(weights, client_states)
            )
            aggregated[name] = weighted_sum.to(client_states[0][name].dtype)

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client round-trip."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # float16 * 2 (upload + download)
```

### `src/federation/aggregators/ffa_lora.py`

```python
"""
FFA-LoRA Aggregator: Freeze A, Aggregate Only B

From Sun et al., ICLR 2024.
Key insight: Freezing A reduces gradient coupling and halves communication.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FFALoRAAggregator:
    """
    FFA-LoRA: Freeze A matrices, only aggregate B.

    - A matrices frozen from initialization
    - Only B matrices are trained and aggregated
    - 50% communication reduction
    """

    def __init__(self):
        self.name = "FFA-LoRA"
        self.frozen_a = None
        self.initialized = False

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate: freeze A, average B.
        """
        if not client_states:
            raise ValueError("No client states")

        # Initialize frozen A on first call
        if not self.initialized:
            self.frozen_a = {}
            for name, param in client_states[0].items():
                if "lora_A" in name or "lora_a" in name:
                    self.frozen_a[name] = param.clone()
            self.initialized = True

        # Default weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total = sum(weights)
        weights = [w / total for w in weights]

        # Aggregate
        aggregated = {}
        for name in client_states[0].keys():
            if "lora_A" in name or "lora_a" in name:
                # Use frozen A
                aggregated[name] = self.frozen_a[name].clone()
            else:
                # Average B (and any other params)
                weighted_sum = sum(
                    w * state[name].float()
                    for w, state in zip(weights, client_states)
                )
                aggregated[name] = weighted_sum.to(client_states[0][name].dtype)

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client (only B matrices after init)."""
        b_params = sum(
            p.numel()
            for name, p in state.items()
            if "lora_B" in name or "lora_b" in name
        )
        return b_params * 2 * 2  # float16 * 2 (upload + download)

    def reset(self):
        """Reset for new experiment."""
        self.frozen_a = None
        self.initialized = False
```

### `src/federation/aggregators/flora.py`

```python
"""
FLoRA Aggregator: Stacking-based Aggregation

From Wang et al., NeurIPS 2024.
Instead of averaging, stack LoRA modules then compress with SVD.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FLoRAAggregator:
    """
    FLoRA: Stack client LoRAs, compress with SVD.

    - Stack A matrices: [A_1; A_2; ... A_K]
    - Concat B matrices: [B_1, B_2, ..., B_K]
    - Compress using SVD to limit size
    """

    def __init__(self, max_rank: int = 64):
        self.name = "FLoRA"
        self.max_rank = max_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stack and compress client LoRAs.
        """
        if not client_states:
            raise ValueError("No client states")

        aggregated = {}

        # Find A/B pairs (PEFT uses lora_A / lora_B)
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            # Find corresponding B key
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            # Stack A matrices (along dim 0): (r, in) -> (K*r, in)
            a_matrices = [s[a_key] for s in client_states]
            stacked_a = torch.cat(a_matrices, dim=0)

            # Concat B matrices (along dim 1): (out, r) -> (out, K*r)
            b_matrices = [s[b_key] for s in client_states]
            stacked_b = torch.cat(b_matrices, dim=1)

            # Compress if needed
            if stacked_a.shape[0] > self.max_rank:
                # Compute BA product (out x in)
                ba = stacked_b.float() @ stacked_a.float()

                # SVD
                U, S, Vh = torch.linalg.svd(ba, full_matrices=False)

                # Truncate
                r = min(self.max_rank, S.shape[0])
                U = U[:, :r]
                S = S[:r]
                Vh = Vh[:r, :]

                # Reconstruct A (r x in), B (out x r)
                sqrt_s = torch.sqrt(S)
                new_b = (U * sqrt_s.unsqueeze(0)).to(b_matrices[0].dtype)
                new_a = (sqrt_s.unsqueeze(1) * Vh).to(a_matrices[0].dtype)

                aggregated[a_key] = new_a
                aggregated[b_key] = new_b
            else:
                aggregated[a_key] = stacked_a
                aggregated[b_key] = stacked_b

        return aggregated

    def get_communication_cost(
        self,
        state: Dict[str, torch.Tensor],
        num_clients: int = 1,
    ) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # Approximate
```

### `src/federation/aggregators/flexlora.py`

```python
"""
FlexLoRA Aggregator: SVD-based Weight Redistribution

From Bai et al., NeurIPS 2024.
Compute full ΔW = Σ(B_k @ A_k), then decompose back to low-rank.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FlexLoRAAggregator:
    """
    FlexLoRA: SVD-based aggregation.

    1. Compute ΔW = Σ w_k * (B_k @ A_k) for all clients
    2. Apply SVD to get low-rank approximation
    3. Redistribute to clients
    """

    def __init__(self, global_rank: int = 32):
        self.name = "FlexLoRA"
        self.global_rank = global_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate using SVD on weight products.
        """
        if not client_states:
            raise ValueError("No client states")

        # Default weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total = sum(weights)
        weights = [w / total for w in weights]

        aggregated = {}

        # Find A/B pairs
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            # Compute weighted sum of BA products
            accumulated = None
            for w, state in zip(weights, client_states):
                a = state[a_key].float()
                b = state[b_key].float()
                update = b @ a

                if accumulated is None:
                    accumulated = w * update
                else:
                    accumulated += w * update

            # SVD decomposition
            U, S, Vh = torch.linalg.svd(accumulated, full_matrices=False)

            # Truncate to global_rank
            r = min(self.global_rank, S.shape[0])
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]

            # Reconstruct A and B
            sqrt_s = torch.sqrt(S)
            new_b = (U * sqrt_s.unsqueeze(0)).to(
                client_states[0][b_key].dtype
            )
            new_a = (sqrt_s.unsqueeze(1) * Vh).to(
                client_states[0][a_key].dtype
            )

            aggregated[a_key] = new_a
            aggregated[b_key] = new_b

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2
```

### `src/federation/aggregators/__init__.py`

```python
"""Federated LoRA aggregation methods."""

from .fedit import FedITAggregator
from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator
from .flexlora import FlexLoRAAggregator

__all__ = [
    "FedITAggregator",
    "FFALoRAAggregator",
    "FLoRAAggregator",
    "FlexLoRAAggregator",
]
```

### Empty stub files (not yet implemented)

These exist but are **empty**:  
`scripts/analyze_results.py`, `scripts/generate_figures.py`, `scripts/run_all_experiments.sh`,  
`src/evaluation/metrics.py`, `src/evaluation/statistical_tests.py`,  
`src/utils/config.py`, `src/utils/logging_utils.py`, `src/utils/memory_utils.py`,  
and `src/**/__init__.py` (package markers).

---

## 7. Known gaps / next implementation tasks

1. **`run_experiment.py` does not yet read** `lora_ranks`, `client_counts`, or loop sweeps — EXP4/EXP5 need driver logic or separate YAMLs per run.
2. **No held-out evaluation** wired in (`eval_fn` is always `None`); `evaluation.eval_every` does nothing useful yet.
3. **FFA-LoRA fidelity**: freezing A usually requires **not training A** client-side; current client trains all LoRA weights unless further constrained.
4. **Communication accounting** in `results.json` is **cumulative** across rounds by design of `server.py`.

---

_End of handoff document._

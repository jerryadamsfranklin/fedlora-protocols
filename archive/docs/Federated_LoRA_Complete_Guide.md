# FEDERATED LORA EXPERIMENTS: COMPLETE GUIDE FOR CURSOR AI

---

# SECTION 0: READ THIS FIRST - WHAT WE ARE BUILDING AND WHY

## 0.1 THE GOAL IN ONE SENTENCE

**We are building a complete experimental framework to compare 4 different methods for combining LoRA adapters in federated learning, so we can publish the results as a research paper.**

---

## 0.2 WHY WE ARE DOING THIS

**The Author's Situation:**
- Jerry Adams is an independent AI/ML researcher
- He needs to publish peer-reviewed research papers for his EB1A immigration application
- He has a Mac M4 Pro with 48GB memory (no GPU cluster access)
- He wants to publish in PeerJ Computer Science (a Q2 journal)

**The Research Gap We Are Filling:**
- Multiple methods exist for federated LoRA fine-tuning (FedIT, FFA-LoRA, FLoRA, FlexLoRA)
- Each paper introducing these methods uses different experimental setups
- **Nobody has systematically compared them under identical conditions**
- This comparison is our contribution to science

**Why This Is Publishable:**
- Original experiments (not a survey of other papers)
- Reproducible (we will release the code)
- Practical (runs on consumer hardware)
- Fills a real gap (no existing systematic comparison)

---

## 0.3 WHAT WE ARE BUILDING (THE DELIVERABLES)

By the end of this project, we will have:

### Deliverable 1: Working Code
```
federated-lora-experiments/
├── src/           # All the Python code
├── config/        # Experiment configurations
├── scripts/       # Scripts to run experiments
└── requirements.txt
```

### Deliverable 2: Experimental Results
```
results/
├── exp1_iid_fedit/results.json
├── exp1_iid_ffa_lora/results.json
├── exp1_iid_flora/results.json
├── exp1_iid_flexlora/results.json
├── exp2_noniid_label_fedit/results.json
├── ... (16+ result files total)
```

### Deliverable 3: Publication Figures
```
figures/
├── fig1_convergence_comparison.pdf
├── fig2_iid_vs_noniid.pdf
├── fig3_communication_tradeoff.pdf
├── fig4_rank_sensitivity.pdf
└── fig5_client_scaling.pdf
```

### Deliverable 4: Research Paper
A complete paper ready for submission to PeerJ Computer Science, with:
- Clear research questions
- Rigorous methodology
- Statistical analysis
- Practical recommendations

---

## 0.4 THE CORE RESEARCH QUESTIONS

Our experiments will answer these specific questions:

| RQ | Question | Why It Matters |
|----|----------|----------------|
| **RQ1** | How do the 4 methods compare when data is evenly distributed (IID)? | Establishes baseline performance |
| **RQ2** | How do they compare when data is unevenly distributed (non-IID)? | Real-world federations always have uneven data |
| **RQ3** | What are the communication costs of each method? | Communication is often the bottleneck |
| **RQ4** | How does LoRA rank affect performance? | Practitioners need to choose a rank |
| **RQ5** | How do methods scale with more clients? | Real federations may have 10-100+ participants |

---

## 0.5 THE 4 METHODS WE ARE COMPARING

| Method | How It Works | Expected Strengths |
|--------|--------------|-------------------|
| **FedIT** | Average all LoRA parameters (A and B matrices) across clients | Simple baseline |
| **FFA-LoRA** | Freeze A matrix, only train and average B | More stable, 50% less communication |
| **FLoRA** | Stack client LoRAs together, compress with SVD | No averaging noise |
| **FlexLoRA** | Compute full weight update, redistribute via SVD | Handles different client capacities |

---

## 0.6 WHAT SUCCESS LOOKS LIKE

The project is successful when:

1. ✅ **Code runs without errors** on Mac M4 with 48GB RAM
2. ✅ **All experiments complete** (approximately 30-40 hours total runtime)
3. ✅ **Results show expected patterns:**
   - Under IID: All methods perform similarly
   - Under non-IID: FedIT degrades most, others are more robust
   - FFA-LoRA uses ~50% of FedIT's communication
4. ✅ **Statistical significance** achieved (p < 0.05 for key comparisons)
5. ✅ **Paper is written** and ready for submission

---

## 0.7 CURSOR AI'S ROLE

**Cursor AI will:**
1. Create the project structure
2. Implement all Python files according to specifications
3. Debug any issues that arise
4. Help run the experiments
5. Generate figures from results

**Cursor AI should:**
- Read this entire document before starting
- Implement files one by one, testing each
- Ask clarifying questions if specifications are unclear
- Optimize for Mac M4 with MPS (Apple Silicon GPU)

---

# SECTION 1: TECHNICAL BACKGROUND

## 1.1 What is Federated Learning?

Federated Learning is a way to train machine learning models without sharing raw data.

**Traditional ML:**
```
All data → Central server → Train model
```

**Federated Learning:**
```
Client 1 data → Client 1 trains locally → Sends model updates to server
Client 2 data → Client 2 trains locally → Sends model updates to server
Client N data → Client N trains locally → Sends model updates to server
                                              ↓
                                    Server aggregates updates
                                              ↓
                                    Sends updated model back to clients
                                              ↓
                                    Repeat for many rounds
```

**Why use it:**
- Data never leaves client devices (privacy)
- Required by regulations (HIPAA, GDPR)
- Clients may be in different organizations

---

## 1.2 What is LoRA?

LoRA (Low-Rank Adaptation) is a method to fine-tune large models efficiently.

**Problem:** LLMs have billions of parameters. Fine-tuning all of them:
- Requires massive GPU memory
- Creates large files to transmit
- Is slow

**LoRA Solution:** Instead of updating all weights W, decompose the update into two small matrices:

```
Original: W_new = W_old + ΔW       (ΔW is huge, e.g., 4096 × 4096)
LoRA:     W_new = W_old + B × A   (B is 4096 × 16, A is 16 × 4096)
```

With rank r=16, this reduces parameters by 256x!

**In code:**
```python
# Instead of storing/transmitting full ΔW (16 million params)
# We store/transmit A (65K params) + B (65K params) = 130K params
```

---

## 1.3 The Problem: How to Aggregate LoRA in Federated Learning?

When multiple clients train LoRA adapters, the server must combine them.

**The naive approach (FedIT):**
```python
global_A = average(client_1_A, client_2_A, ..., client_N_A)
global_B = average(client_1_B, client_2_B, ..., client_N_B)
```

**The issue:** This doesn't account for the fact that what matters is the PRODUCT B×A, not A and B separately. Averaging them independently introduces noise.

**Better approaches try to fix this:**
- FFA-LoRA: Don't average A at all (freeze it)
- FLoRA: Stack the matrices instead of averaging
- FlexLoRA: Compute B×A first, then decompose

---

## 1.4 What Our Experiments Will Show

We will run controlled experiments to measure:

1. **Final Model Quality** (lower loss = better)
   - Which method produces the best model after 30 rounds?

2. **Convergence Speed** (faster = better)
   - Which method reaches good performance quickest?

3. **Communication Cost** (lower = better)
   - Which method transmits the fewest bytes?

4. **Robustness to Heterogeneity** (less degradation = better)
   - Which method handles non-IID data best?

---

# SECTION 2: SYSTEM REQUIREMENTS AND SETUP

## 2.1 Hardware Requirements

**Minimum:**
- Mac M1/M2/M3/M4 with 16GB unified memory
- 50GB free disk space

**Recommended (what Jerry has):**
- Mac M4 Pro with 48GB unified memory
- 100GB free disk space
- Fast SSD

**What we can run with 48GB:**
- TinyLlama-1.1B: Full precision ✓
- LLaMA-3.2-3B: Full precision ✓
- Mistral-7B: 4-bit quantized ✓
- Up to 20 simulated clients ✓

---

## 2.2 Software Setup

**Step 1: Install Miniforge (Conda for Apple Silicon)**
```bash
brew install miniforge
conda init zsh
# Restart terminal
```

**Step 2: Create Project Directory**
```bash
mkdir -p ~/research/federated-lora-experiments
cd ~/research/federated-lora-experiments
```

**Step 3: Create Conda Environment**
```bash
conda create -n fedlora python=3.10 -y
conda activate fedlora
```

**Step 4: Install Dependencies**
```bash
# PyTorch with MPS support
pip install torch==2.2.0 torchvision torchaudio

# Hugging Face ecosystem
pip install transformers==4.40.0
pip install accelerate==0.29.0
pip install peft==0.10.0
pip install datasets==2.18.0
pip install evaluate==0.4.1

# Scientific computing
pip install numpy==1.26.0
pip install scipy==1.12.0
pip install scikit-learn==1.4.0
pip install pandas==2.2.0

# Visualization
pip install matplotlib==3.8.0
pip install seaborn==0.13.0

# Utilities
pip install pyyaml==6.0.1
pip install tqdm==4.66.0
pip install wandb==0.16.0  # Optional
```

**Step 5: Verify MPS Works**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Should print: MPS available: True
```

**Step 6: Login to Hugging Face**
```bash
pip install huggingface_hub
huggingface-cli login
# Enter token from https://huggingface.co/settings/tokens
```

---

## 2.3 Project Structure to Create

```
federated-lora-experiments/
│
├── README.md
├── requirements.txt
│
├── config/
│   ├── base_config.yaml
│   ├── exp1_iid.yaml
│   ├── exp2_noniid_label.yaml
│   ├── exp3_noniid_quantity.yaml
│   ├── exp4_rank_sensitivity.yaml
│   └── exp5_client_scaling.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── lora_model.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_partitioner.py
│   │
│   ├── federation/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── aggregators/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── fedit.py
│   │       ├── ffa_lora.py
│   │       ├── flora.py
│   │       └── flexlora.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── statistical_tests.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       └── memory_utils.py
│
├── scripts/
│   ├── run_experiment.py
│   ├── run_all_experiments.sh
│   ├── analyze_results.py
│   └── generate_figures.py
│
├── results/
│   └── .gitkeep
│
└── figures/
    └── .gitkeep
```

---

# SECTION 3: DETAILED SPECIFICATIONS FOR EACH FILE

## 3.1 Configuration Files

### File: config/base_config.yaml

**Purpose:** Default settings inherited by all experiments.

```yaml
# =============================================================================
# BASE CONFIGURATION
# All experiment configs inherit from this and can override specific values
# =============================================================================

# Model Configuration
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Primary model for experiments
  # Alternative: "meta-llama/Llama-3.2-3B" for larger experiments
  torch_dtype: "float16"
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

### File: config/exp1_iid.yaml

**Purpose:** Experiment 1 - IID baseline comparison.

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

### File: config/exp2_noniid_label.yaml

**Purpose:** Experiment 2 - Non-IID with label distribution skew.

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

### File: config/exp3_noniid_quantity.yaml

**Purpose:** Experiment 3 - Non-IID with quantity skew.

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

### File: config/exp4_rank_sensitivity.yaml

**Purpose:** Experiment 4 - How does LoRA rank affect results?

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

### File: config/exp5_client_scaling.yaml

**Purpose:** Experiment 5 - How does number of clients affect results?

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

---

## 3.2 Core Source Files

### File: src/models/lora_model.py

**PURPOSE:** Wrap a HuggingFace model with LoRA adapters for federated learning.

**WHAT IT DOES:**
1. Loads a pre-trained LLM (TinyLlama, LLaMA, etc.)
2. Applies LoRA adapters to specified layers
3. Provides methods to extract/load only LoRA parameters
4. Manages memory on Apple Silicon (MPS)

**KEY METHODS:**
- `load_model()`: Load the model and apply LoRA
- `get_lora_state_dict()`: Extract only LoRA params (for sending to server)
- `set_lora_state_dict()`: Load LoRA params (received from server)
- `get_communication_cost_bytes()`: Calculate bytes to transmit

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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, List, Optional
import gc


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
        device: str = "mps"
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
        
        self._model = None
        self._tokenizer = None
    
    def load_model(self) -> None:
        """Load base model and apply LoRA adapters."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self._model = get_peft_model(base_model, lora_config)
        self._model.print_trainable_parameters()
    
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

---

### File: src/data/data_partitioner.py

**PURPOSE:** Split a dataset among multiple clients with IID or non-IID distributions.

**WHAT IT DOES:**
1. Takes a HuggingFace dataset and number of clients
2. Splits data according to specified strategy:
   - IID: Random uniform split
   - Label skew: Different clients get different label distributions
   - Quantity skew: Different clients get different amounts of data

**KEY METHODS:**
- `iid_partition()`: Random uniform split
- `label_skew_partition(alpha)`: Dirichlet-based label distribution
- `quantity_skew_partition(alpha)`: Dirichlet-based quantity distribution

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

import numpy as np
from typing import List, Dict, Optional
from datasets import Dataset
from collections import defaultdict


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
        alpha: float = 0.5
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
                client_indices[client_id].extend(indices[start:start+count])
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
        min_samples: int = 10
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
        extra = (proportions * remaining).astype(int)
        extra[-1] = remaining - extra[:-1].sum()
        
        sizes = base + extra
        
        # Shuffle and split
        indices = np.random.permutation(total)
        client_datasets = []
        start = 0
        for size in sizes:
            client_datasets.append(
                self.dataset.select(indices[start:start+size].tolist())
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
            "mean_samples": np.mean(sizes),
            "std_samples": np.std(sizes)
        }
```

---

### File: src/federation/client.py

**PURPOSE:** Simulate a federated learning client that trains locally.

**WHAT IT DOES:**
1. Receives global LoRA state from server
2. Trains on local data for specified epochs
3. Returns updated LoRA state and training metrics

**KEY METHODS:**
- `train(global_state)`: Perform local training, return updated state

```python
"""
Federated Learning Client

Simulates a client in federated learning:
1. Receives global model state from server
2. Trains on local data
3. Returns updated model state

CURSOR AI: Implement this file with all methods as specified.
"""

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from typing import Dict, Optional
from tqdm import tqdm
import time


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
        gradient_accumulation_steps: int = 4
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
                    for inst, out in zip(examples["instruction"], examples["output"])
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
                padding="max_length"
            )
        
        # Tokenize
        tokenized = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        tokenized.set_format("torch")
        
        # Create dataloader
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False
        )
        self.dataloader = DataLoader(
            tokenized,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator
        )
    
    def train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None
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
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        self.model.model.train()
        total_loss = 0
        num_steps = 0
        start_time = time.time()
        
        for epoch in range(self.local_epochs):
            for step, batch in enumerate(tqdm(
                self.dataloader,
                desc=f"Client {self.client_id} Epoch {epoch+1}",
                leave=False
            )):
                # Move to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Forward
                outputs = self.model.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps
                
                # Backward
                loss.backward()
                
                # Update
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_steps += 1
        
        training_time = time.time() - start_time
        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        
        # Get updated state
        updated_state = self.model.get_lora_state_dict()
        
        # Cleanup
        optimizer.zero_grad(set_to_none=True)
        self.model.clear_memory()
        
        return {
            "state_dict": updated_state,
            "loss": avg_loss,
            "num_samples": len(self.dataset),
            "training_time": training_time
        }
```

---

### File: src/federation/aggregators/fedit.py

**PURPOSE:** Implement FedIT aggregation (baseline FedAvg on LoRA).

**WHAT IT DOES:**
1. Takes list of client LoRA states
2. Computes weighted average of each parameter
3. Returns aggregated global state

```python
"""
FedIT Aggregator: Baseline FedAvg on LoRA

Simply averages A and B matrices independently across clients.
This is the baseline method from Zhang et al., ICASSP 2024.

CURSOR AI: Implement this file as specified.
"""

import torch
from typing import List, Dict, Optional


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
        weights: Optional[List[float]] = None
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

---

### File: src/federation/aggregators/ffa_lora.py

**PURPOSE:** Implement FFA-LoRA (Freeze A, aggregate only B).

```python
"""
FFA-LoRA Aggregator: Freeze A, Aggregate Only B

From Sun et al., ICLR 2024.
Key insight: Freezing A reduces gradient coupling and halves communication.

CURSOR AI: Implement this file as specified.
"""

import torch
from typing import List, Dict, Optional


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
        weights: Optional[List[float]] = None
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
            p.numel() for name, p in state.items()
            if "lora_B" in name or "lora_b" in name
        )
        return b_params * 2 * 2  # float16 * 2 (upload + download)
    
    def reset(self):
        """Reset for new experiment."""
        self.frozen_a = None
        self.initialized = False
```

---

### File: src/federation/aggregators/flora.py

**PURPOSE:** Implement FLoRA (stacking-based aggregation).

```python
"""
FLoRA Aggregator: Stacking-based Aggregation

From Wang et al., NeurIPS 2024.
Instead of averaging, stack LoRA modules then compress with SVD.

CURSOR AI: Implement this file as specified.
"""

import torch
from typing import List, Dict, Optional


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
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Stack and compress client LoRAs.
        """
        if not client_states:
            raise ValueError("No client states")
        
        aggregated = {}
        
        # Find A/B pairs
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]
        
        for a_key in a_keys:
            # Find corresponding B key
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue
            
            # Stack A matrices (along dim 0)
            a_matrices = [s[a_key] for s in client_states]
            stacked_a = torch.cat(a_matrices, dim=0)
            
            # Concat B matrices (along dim 1)
            b_matrices = [s[b_key] for s in client_states]
            stacked_b = torch.cat(b_matrices, dim=1)
            
            # Compress if needed
            if stacked_a.shape[0] > self.max_rank:
                # Compute BA product
                ba = stacked_b.float() @ stacked_a.float()
                
                # SVD
                U, S, Vh = torch.linalg.svd(ba, full_matrices=False)
                
                # Truncate
                U = U[:, :self.max_rank]
                S = S[:self.max_rank]
                Vh = Vh[:self.max_rank, :]
                
                # Reconstruct
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
        num_clients: int = 1
    ) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # Approximate
```

---

### File: src/federation/aggregators/flexlora.py

**PURPOSE:** Implement FlexLoRA (SVD-based redistribution).

```python
"""
FlexLoRA Aggregator: SVD-based Weight Redistribution

From Bai et al., NeurIPS 2024.
Compute full ΔW = Σ(B_k @ A_k), then decompose back to low-rank.

CURSOR AI: Implement this file as specified.
"""

import torch
from typing import List, Dict, Optional


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
        weights: Optional[List[float]] = None
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
            U = U[:, :self.global_rank]
            S = S[:self.global_rank]
            Vh = Vh[:self.global_rank, :]
            
            # Reconstruct A and B
            sqrt_s = torch.sqrt(S)
            new_b = (U * sqrt_s.unsqueeze(0)).to(client_states[0][b_key].dtype)
            new_a = (sqrt_s.unsqueeze(1) * Vh).to(client_states[0][a_key].dtype)
            
            aggregated[a_key] = new_a
            aggregated[b_key] = new_b
        
        return aggregated
    
    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2
```

---

### File: src/federation/server.py

**PURPOSE:** Coordinate federated training across clients.

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

import os
import json
import time
import torch
from typing import List, Dict, Optional
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
        "flexlora": FlexLoRAAggregator
    }
    
    def __init__(
        self,
        aggregation_method: str = "fedit",
        num_rounds: int = 30,
        eval_every: int = 5,
        output_dir: str = "results"
    ):
        self.num_rounds = num_rounds
        self.eval_every = eval_every
        self.output_dir = output_dir
        
        # Initialize aggregator
        if aggregation_method not in self.AGGREGATORS:
            raise ValueError(f"Unknown method: {aggregation_method}")
        self.aggregator = self.AGGREGATORS[aggregation_method]()
        
        self.global_state = None
        self.clients = []
        self.metrics_history = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def set_clients(self, clients: List):
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
                
                # Track communication
                total_communication += sum(
                    p.numel() * 2 for p in result["state_dict"].values()
                )
            
            # Aggregate
            self.global_state = self.aggregator.aggregate(
                client_states,
                weights=client_weights
            )
            
            # Add download communication
            total_communication += sum(
                p.numel() * 2 for p in self.global_state.values()
            ) * len(self.clients)
            
            round_time = time.time() - round_start
            avg_loss = sum(losses) / len(losses)
            
            # Log
            metrics = {
                "round": round_num + 1,
                "avg_loss": avg_loss,
                "round_time": round_time,
                "communication_mb": total_communication / (1024 * 1024)
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
            "total_communication_mb": total_communication / (1024 * 1024)
        }
    
    def _save_results(self):
        """Save results to JSON."""
        path = os.path.join(self.output_dir, "results.json")
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"\nResults saved to {path}")
```

---

## 3.3 Scripts

### File: scripts/run_experiment.py

**PURPOSE:** Main script to run a single experiment.

```python
"""
Run a single federated LoRA experiment.

Usage:
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit

CURSOR AI: Implement this script as specified.
"""

import argparse
import yaml
import os
import sys
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lora_model import FederatedLoRAModel
from src.data.data_partitioner import DataPartitioner
from src.federation.client import FederatedClient
from src.federation.server import FederatedServer
from datasets import load_dataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--method", default=None, help="Override aggregation method")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override method if specified
    method = args.method or config.get("federated", {}).get("aggregation_method", "fedit")
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{config['experiment']['name']}_{method}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n[1/4] Loading model...")
    model = FederatedLoRAModel(
        model_name=config.get("model", {}).get("name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        lora_r=config.get("lora", {}).get("r", 16),
        lora_alpha=config.get("lora", {}).get("lora_alpha", 32),
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.load_model()
    
    # Load data
    print("\n[2/4] Loading data...")
    dataset = load_dataset(
        config["data"]["dataset_name"],
        split=config["data"]["dataset_split"]
    )
    
    if "max_samples" in config["data"]:
        dataset = dataset.select(range(min(config["data"]["max_samples"], len(dataset))))
    
    # Partition
    partitioner = DataPartitioner(
        dataset,
        num_clients=config.get("federated", {}).get("num_clients", 10),
        seed=args.seed
    )
    
    partition_method = config["data"].get("partition_method", "iid")
    if partition_method == "iid":
        client_datasets = partitioner.iid_partition()
    elif partition_method == "label_skew":
        client_datasets = partitioner.label_skew_partition(
            label_column=config["data"].get("label_column", "label"),
            alpha=config["data"].get("dirichlet_alpha", 0.5)
        )
    elif partition_method == "quantity_skew":
        client_datasets = partitioner.quantity_skew_partition(
            alpha=config["data"].get("quantity_alpha", 0.5)
        )
    else:
        client_datasets = partitioner.iid_partition()
    
    print(f"Partition stats: {partitioner.get_stats(client_datasets)}")
    
    # Create clients
    print("\n[3/4] Creating clients...")
    clients = []
    for i, ds in enumerate(client_datasets):
        client = FederatedClient(
            client_id=i,
            model=model,
            dataset=ds,
            batch_size=config.get("training", {}).get("batch_size", 4),
            local_epochs=config.get("training", {}).get("local_epochs", 2),
            learning_rate=config.get("training", {}).get("learning_rate", 2e-4)
        )
        clients.append(client)
    
    # Create server
    server = FederatedServer(
        aggregation_method=method,
        num_rounds=config.get("federated", {}).get("num_rounds", 30),
        eval_every=config.get("evaluation", {}).get("eval_every", 5),
        output_dir=output_dir
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

---

# SECTION 4: RUNNING THE EXPERIMENTS

## 4.1 Order of Execution

```bash
# Activate environment
conda activate fedlora
cd ~/research/federated-lora-experiments

# Test with minimal config first
python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit

# If successful, run all methods for EXP1
for method in fedit ffa_lora flora flexlora; do
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method $method
done

# Then EXP2
for method in fedit ffa_lora flora flexlora; do
    python scripts/run_experiment.py --config config/exp2_noniid_label.yaml --method $method
done

# Continue with EXP3, EXP4, EXP5...
```

## 4.2 Expected Runtime

| Experiment | Methods | Est. Time |
|------------|---------|-----------|
| EXP1-IID | 4 | ~6-8 hours |
| EXP2-NonIID-Label | 4 | ~6-8 hours |
| EXP3-NonIID-Quantity | 4 | ~6-8 hours |
| EXP4-Rank | 10 runs | ~8-10 hours |
| EXP5-Scale | 6 runs | ~4-6 hours |
| **TOTAL** | - | **~30-40 hours** |

---

# SECTION 5: EXPECTED RESULTS

## 5.1 What We Expect to Find

### EXP1 (IID):
- All methods achieve similar final loss
- FLoRA/FlexLoRA converge slightly faster
- FFA-LoRA uses ~50% communication of FedIT

### EXP2 (Non-IID Label):
- FedIT shows 10-20% performance degradation vs IID
- FFA-LoRA shows 5-10% degradation
- FLoRA/FlexLoRA show <5% degradation

### EXP3 (Non-IID Quantity):
- Higher variance across runs
- FlexLoRA handles imbalance best

### EXP4 (Rank):
- Performance improves from r=4 to r=32
- Diminishing returns after r=32
- Communication scales linearly

### EXP5 (Scaling):
- All methods degrade with more clients
- FLoRA communication grows most

---

# SECTION 6: PAPER STRUCTURE

## Title
"Empirical Comparison of Federated LoRA Aggregation Strategies on Consumer Hardware"

## Sections
1. Introduction (motivation, gap, contributions)
2. Background (FL, LoRA, existing methods)
3. Methodology (setup, datasets, metrics)
4. Results (RQ1-RQ5 findings)
5. Discussion (recommendations, limitations)
6. Related Work
7. Conclusion

## Key Claims
1. First systematic comparison on identical setup
2. Reproducible on consumer hardware
3. Practical recommendations for practitioners

---

# SECTION 7: CHECKLIST FOR CURSOR AI

## Before Starting
- [ ] Read this entire document
- [ ] Understand the research goals
- [ ] Understand the 4 methods

## Implementation Phase
- [ ] Create directory structure
- [ ] Implement src/models/lora_model.py
- [ ] Implement src/data/data_partitioner.py
- [ ] Implement src/federation/client.py
- [ ] Implement all 4 aggregators
- [ ] Implement src/federation/server.py
- [ ] Implement scripts/run_experiment.py
- [ ] Test with minimal config

## Experiment Phase
- [ ] Run EXP1 (all methods)
- [ ] Run EXP2 (all methods)
- [ ] Run EXP3 (all methods)
- [ ] Run EXP4 (rank sensitivity)
- [ ] Run EXP5 (client scaling)

## Analysis Phase
- [ ] Generate convergence figures
- [ ] Generate comparison tables
- [ ] Compute statistical tests
- [ ] Write results summary

---

**END OF DOCUMENT**

This document contains everything needed to implement and run the experiments.
Cursor AI should read this entire document before beginning.

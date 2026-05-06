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
        lora_param_dtype: Optional[Union[str, torch.dtype]] = None,
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
        self.lora_param_dtype = lora_param_dtype

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

        # Load base model.
        #
        # On MPS, loading large sharded checkpoints directly to device via
        # device_map can hit dtype edge cases during shard materialization.
        # We load on CPU first, then move to MPS after weights are instantiated.
        load_device_map = {"": self.device}
        if self.device == "mps":
            load_device_map = {"": "cpu"}

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=load_device_map,
            trust_remote_code=True,
        )
        if self.device == "mps":
            base_model = base_model.to(self.device)
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

        if self.lora_param_dtype is not None:
            lp_dtype = self._resolve_dtype(self.lora_param_dtype)
            self._cast_trainable_lora_params(lp_dtype)
            print(f"lora_param_dtype (train): {lp_dtype}")

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

    def _cast_trainable_lora_params(self, dtype: torch.dtype) -> None:
        """Cast trainable LoRA adapter weights to dtype (e.g. fp32 on fp16 base)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    param.data = param.data.to(dtype=dtype)

    def _upload_dtype(self) -> torch.dtype:
        """Dtype used when exporting LoRA tensors for aggregation (matches base model)."""
        return self._resolve_dtype(self.torch_dtype)

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
        upload_dtype = self._upload_dtype()
        state = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name and param.requires_grad:
                t = param.detach().to(dtype=upload_dtype)
                state[name] = t.cpu().clone()
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
                dst = current_state[name]
                src = param.to(device=self.device, dtype=dst.dtype)
                dst.copy_(src)

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

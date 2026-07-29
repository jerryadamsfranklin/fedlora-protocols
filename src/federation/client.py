"""
Federated Learning Client

Simulates a client in federated learning:
1. Receives global model state from server
2. Trains on local data
3. Returns updated model state

CURSOR AI: Implement this file with all methods as specified.
"""

import time
from typing import Dict, List, Optional

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
        def _format_commonsenseqa(examples):
            questions = examples["question"]
            choices = examples["choices"]
            answer_keys = examples.get("answerKey")

            formatted = []
            for i, q in enumerate(questions):
                # `choices` can appear as either:
                # - a dict-of-lists (batched): {"label": [[...]], "text": [[...]]}
                # - a list-of-dicts: [{"label": [...], "text": [...]}, ...]
                if isinstance(choices, dict):
                    labels = choices["label"][i]
                    texts = choices["text"][i]
                else:
                    labels = choices[i]["label"]
                    texts = choices[i]["text"]

                choices_lines = "\n".join(
                    f"{lab}) {txt}" for lab, txt in zip(labels, texts)
                )
                ans = answer_keys[i] if answer_keys is not None else ""
                formatted.append(
                    f"Question: {q}\n\nChoices:\n{choices_lines}\n\nAnswer: {ans}"
                )
            return formatted

        def tokenize(examples):
            # Handle different dataset formats
            if "instruction" in examples:
                outputs = examples.get("output")
                responses = examples.get("response")
                contexts = examples.get("context")
                texts = []
                for i, inst in enumerate(examples["instruction"]):
                    out = ""
                    if outputs is not None:
                        out = outputs[i]
                    elif responses is not None:
                        out = responses[i]
                    ctx = contexts[i] if contexts is not None else ""
                    if ctx:
                        text = (
                            f"### Instruction:\n{inst}\n\n"
                            f"### Context:\n{ctx}\n\n"
                            f"### Response:\n{out}"
                        )
                    else:
                        text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    texts.append(text)
            elif "question" in examples:
                # CommonsenseQA: include choices + answer for a meaningful task.
                if "choices" in examples and "answerKey" in examples:
                    texts = _format_commonsenseqa(examples)
                else:
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

        sample = next(iter(self.dataloader))
        seq_len = sample["input_ids"].shape[1]
        assert (
            seq_len == self.max_seq_length
        ), f"Expected seq_len {self.max_seq_length}, got {seq_len}"

    def train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        freeze_a: bool = False,
        freeze_ratio: float = 0.0,
        freeze_ratios: Optional[Dict[str, float]] = None,
        current_rank: Optional[int] = None,
        collect_layer_metrics: bool = False,
        b_only_upload: bool = False,
    ) -> Dict:
        """
        Perform local training.

        Args:
            global_state: LoRA state from server (None for first round)
            freeze_a: DEPRECATED - kept for backwards compatibility
            freeze_ratio: Fraction of A matrices to freeze (0.0 to 1.0)
                         - 1.0: Freeze all A (pure FFA-LoRA)
                         - 0.0: Freeze none (pure FLoRA)
                         - 0.5: Freeze half the A matrices
            freeze_ratios: Optional per-layer freeze ratios (layer_name -> [0,1]).
                          If provided, overrides freeze_ratio/freeze_a.
            collect_layer_metrics: If True, returns per-layer grad-norm metrics.

        Returns:
            Dict with:
            - state_dict: Updated LoRA parameters
            - loss: Average training loss
            - num_samples: Number of samples trained on
            - training_time: Time taken in seconds
            - layer_metrics: Optional per-layer metrics (if collect_layer_metrics)
        """
        # Load global state if provided
        if global_state is not None:
            self.model.set_lora_state_dict(global_state)

        # Curriculum-rank: constrain training to the leading `current_rank`
        # dimensions of LoRA A/B. We do this by masking gradients after backward
        # (no persistent hooks), then slicing the state dict for upload.
        if current_rank is not None:
            current_rank = int(current_rank)

        # Apply per-layer freezing if provided; otherwise use global freezing knobs.
        if freeze_ratios is not None:
            self._apply_per_layer_freeze(freeze_ratios)
        else:
            # Backwards compatibility: `freeze_a=True` implies fully frozen A.
            if freeze_a:
                freeze_ratio = 1.0

            freeze_ratio = float(freeze_ratio)
            if freeze_ratio < 0.0:
                freeze_ratio = 0.0
            if freeze_ratio > 1.0:
                freeze_ratio = 1.0

            # Apply partial A freezing (deterministic by parameter name order).
            if freeze_ratio > 0.0:
                self._apply_partial_freeze(freeze_ratio)

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

        # Track gradient norms per layer
        layer_grad_norms: Dict[str, List[float]] = {}

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

                if current_rank is not None:
                    self._mask_lora_grads(current_rank)

                # Collect per-layer grad norms before optimizer step
                if collect_layer_metrics:
                    self._collect_grad_norms(layer_grad_norms)

                # Update
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                num_steps += 1

        training_time = time.time() - start_time
        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0

        # After training, unfreeze everything for clean state.
        self._unfreeze_all_lora()

        avg_layer_metrics: Optional[Dict[str, float]] = None
        if collect_layer_metrics:
            avg_layer_metrics = {
                layer: (sum(vals) / len(vals) if vals else 0.0)
                for layer, vals in layer_grad_norms.items()
            }

        # Get updated state
        updated_state = self.model.get_lora_state_dict()
        if current_rank is not None:
            updated_state = self._slice_lora_state(updated_state, int(current_rank))

        # Communication optimization: optionally upload only LoRA-B tensors.
        # Used for FFA-LoRA / Two-Phase phase-2 after frozen A has been initialized server-side.
        upload_state = updated_state
        if b_only_upload:
            upload_state = self._filter_b_only(updated_state)

        # Cleanup
        optimizer.zero_grad(set_to_none=True)
        self.model.clear_memory()

        result = {
            "state_dict": upload_state,
            "loss": avg_loss,
            "num_samples": len(self.dataset),
            "training_time": training_time,
        }
        if avg_layer_metrics is not None:
            result["layer_metrics"] = avg_layer_metrics
        return result

    def _mask_lora_grads(self, rank: int) -> None:
        """Zero gradient outside the leading `rank` LoRA dimensions."""
        rank = int(rank)
        for name, param in self.model.model.named_parameters():
            if param.grad is None:
                continue
            lname = name.lower()
            if "lora_a" in lname and param.grad.ndim >= 2 and param.grad.shape[0] > rank:
                param.grad[rank:, :] = 0
            elif "lora_b" in lname and param.grad.ndim >= 2 and param.grad.shape[1] > rank:
                param.grad[:, rank:] = 0

    @staticmethod
    def _slice_lora_state(
        state: Dict[str, torch.Tensor],
        rank: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Slice LoRA tensors down to `rank` for upload.

        - A: keep first `rank` rows
        - B: keep first `rank` cols
        """
        rank = int(rank)
        out: Dict[str, torch.Tensor] = {}
        for k, t in state.items():
            lk = k.lower()
            if "lora_a" in lk and t.ndim >= 2 and t.shape[0] > rank:
                out[k] = t[:rank, :].contiguous()
            elif "lora_b" in lk and t.ndim >= 2 and t.shape[1] > rank:
                out[k] = t[:, :rank].contiguous()
            else:
                out[k] = t
        return out

    @staticmethod
    def _filter_b_only(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter LoRA state dict to B matrices only for communication-efficient upload.
        """
        return {k: v for k, v in state.items() if ("lora_B" in k or "lora_b" in k)}

    def _apply_partial_freeze(self, freeze_ratio: float) -> None:
        """
        Freeze a fraction of LoRA A parameters.

        Strategy: freeze the first `freeze_ratio` fraction of A params by
        deterministic layer/parameter name order.
        """
        a_params = [
            (name, param)
            for name, param in self.model.model.named_parameters()
            if ("lora_A" in name or "lora_a" in name)
        ]
        a_params.sort(key=lambda x: x[0])

        num_to_freeze = int(len(a_params) * float(freeze_ratio))

        for i, (_name, param) in enumerate(a_params):
            param.requires_grad = i >= num_to_freeze

    def _unfreeze_all_lora(self) -> None:
        """Unfreeze all LoRA parameters."""
        for name, param in self.model.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    def _extract_layer_id(self, param_name: str) -> str:
        """
        Extract layer identifier from parameter name.

        Examples:
            "...q_proj.lora_A.weight" -> "q_proj"
            "...mlp.gate_proj.lora_A.weight" -> "gate_proj"
        """
        parts = param_name.split(".")
        for i, part in enumerate(parts):
            if "lora" in part.lower():
                if i > 0:
                    return parts[i - 1]
        return "unknown"

    def _apply_per_layer_freeze(self, freeze_ratios: Dict[str, float]) -> None:
        """
        Apply per-layer freezing of LoRA A parameters.

        For each layer_id, freeze the first `ratio` fraction of that layer's
        LoRA-A parameters by deterministic name order.
        """
        # Collect LoRA-A params grouped by layer id
        grouped: Dict[str, List[tuple]] = {}
        for name, param in self.model.model.named_parameters():
            if ("lora_A" in name or "lora_a" in name):
                layer_id = self._extract_layer_id(name)
                grouped.setdefault(layer_id, []).append((name, param))

        for layer_id, params in grouped.items():
            ratio = float(freeze_ratios.get(layer_id, 0.0))
            ratio = max(0.0, min(1.0, ratio))

            params.sort(key=lambda x: x[0])
            num_to_freeze = int(len(params) * ratio)
            for i, (_name, param) in enumerate(params):
                param.requires_grad = i >= num_to_freeze

    def _collect_grad_norms(self, layer_grad_norms: Dict[str, List[float]]) -> None:
        """Collect gradient norms for each LoRA layer (A and B combined)."""
        for name, param in self.model.model.named_parameters():
            if "lora" not in name.lower():
                continue
            if param.grad is None:
                continue
            layer_id = self._extract_layer_id(name)
            layer_grad_norms.setdefault(layer_id, []).append(
                float(param.grad.norm().detach().cpu().item())
            )

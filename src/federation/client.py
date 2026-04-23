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
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(
                        examples["instruction"], examples["output"]
                    )
                ]
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

    def train(
        self,
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        freeze_a: bool = False,
    ) -> Dict:
        """
        Perform local training.

        Args:
            global_state: LoRA state from server (None for first round)
            freeze_a: If True, do not train lora_A (FFA-LoRA protocol)

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

        # FFA-LoRA: keep A frozen during local optimization (train B only)
        for name, param in self.model.model.named_parameters():
            if "lora_A" in name or "lora_a" in name:
                param.requires_grad = not freeze_a

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

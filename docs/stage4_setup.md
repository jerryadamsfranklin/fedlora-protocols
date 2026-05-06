## Stage 4 setup: gated LLaMA-3.2-3B access

Stage 4 uses the gated HuggingFace model `meta-llama/Llama-3.2-3B`.

You must:

1. **Accept the license** on HuggingFace:
   - `https://huggingface.co/meta-llama/Llama-3.2-3B`
2. **Login** locally (one-time per machine/account):

```bash
huggingface-cli login
```

Use a token that has access to the gated model.

Notes:
- Do not attempt to bypass gating. If you do not have access, model download will fail.
- If downloads fail after login, verify the token scopes and that the license was accepted.
- **MPS / fp16 stability**: Stage 4 configs keep the frozen base model in fp16 for memory, but set `training.lora_param_dtype: "float32"` so **LoRA adapters train in fp32**. Uploaded adapter tensors are cast back to the base dtype for aggregation (communication math unchanged).


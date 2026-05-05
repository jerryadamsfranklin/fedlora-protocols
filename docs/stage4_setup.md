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


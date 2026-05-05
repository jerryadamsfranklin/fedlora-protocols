#!/bin/bash
# Memory smoke test for LLaMA-3.2-3B. Runs 1 round, 2 clients only.

set -e
python3 scripts/run_experiment.py \
    --config config/exp_llama3_flora_iid.yaml \
    --override federated.num_rounds=1 \
    --override federated.num_clients=2 \
    --override federated.clients_per_round=2 \
    --seed 42 \
    --tag llama3_smoke

echo "Smoke run complete. Check that peak memory was under 40 GB."


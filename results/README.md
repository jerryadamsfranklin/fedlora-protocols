## Results Tree Scope

This `results/` tree contains the full development history of experiments, including pilot runs, smoke tests, and runs produced at earlier code revisions.

Only runs listed in `results/MANIFEST.csv` back reported paper results. Treat any run not listed there as non-reporting history.

## Hazards When Aggregating Naively

1. Four-target duplicates of the Dolly and baseline experiments are present alongside q/v-only runs. These differ from the reported runs by exactly 2x communication and can make a naive aggregate suggest the paper understated communication by half.
2. Some Two-Phase runs recording `2578.125` MB predate the bidirectional B-only implementation and never switched; they are indistinguishable from FLoRA totals unless filtered by manifest membership.

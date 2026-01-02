#!/bin/sh

# GPTJ, orig dataset
uv run python structrels/sweep.py \
  --experiment-name gptj-sweep-orig \
  --results-dir results \
  --model gptj \
  --recall-k 1 \
  --load-data-from data/orig \
  --limit-test-samples 50

# LLAMA31, orig dataset
# uv run python structrels/sweep.py \
#   --experiment-name llama31-sweep-orig \
#   --results-dir results \
#   --model llama31 \
#   --recall-k 1 \
#   --load_data_from data/orig \
#   --limit-test-samples 50

# NEOX, orig dataset
# uv run python structrels/sweep.py \
#   --experiment-name neox-sweep-orig \
#   --results-dir results \
#   --model neox \
#   --recall-k 1 \
#   --load_data_from data/orig \
#   --limit-test-samples 50


# GPTJ, extended dataset
# uv run python structrels/sweep.py \
#   --experiment-name gptj-sweep-extended \
#   --results-dir results \
#   --model gptj \
#   --recall-k 1 \
#   --load-data-from data/extended \
#   --limit-test-samples 50

# LLAMA31, extended dataset
# uv run python structrels/sweep.py \
#   --experiment-name llama31-sweep-extended \
#   --results-dir results \
#   --model llama31 \
#   --recall-k 1 \
#   --load_data_from data/extended \
#   --limit-test-samples 50

# NEOX, extended dataset
# uv run python structrels/sweep.py \
#   --experiment-name neox-sweep-extended \
#   --results-dir results \
#   --model neox \
#   --recall-k 1 \
#   --load_data_from data/extended \
#   --limit-test-samples 50

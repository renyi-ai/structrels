#!/bin/sh

# GPTJ, orig dataset - test on a small subset (resources/cross_eval_shortlist_test.txt)
uv run python structrels/cross_eval.py \
  --model gptj \
  --data_path data/orig \
  --num_samples 50 \
  --decoder_folder results/gptj-sweep-orig/matrices_hstacked \
  --hparams_path hparams/gptj \
  --result_prefix cross-eval-orig-gptj \
  --shortlist_path scripts/resources/cross_eval_shortlist_test.txt

# GPTJ, orig dataset
# uv run python structrels/cross_eval.py \
#   --model gptj \
#   --data_path data/orig \
#   --num_samples 50 \
#   --decoder_folder results/gptj-sweep-orig/matrices_hstacked \
#   --hparams_path hparams/gptj \
#   --result_prefix cross-eval-orig-gptj

# NEOX, orig dataset
# uv run python structrels/cross_eval.py \
#   --model neox \
#   --data_path data/orig \
#   --num_samples 50 \
#   --decoder_folder results/neox-sweep-orig/matrices_hstacked \
#   --hparams_path hparams/neox \
#   --result_prefix cross-eval-orig-neox

# LLAMA31, orig dataset
# uv run python structrels/cross_eval.py \
#   --model llama31 \
#   --data_path data/orig \
#   --num_samples 50 \
#   --decoder_folder results/llama31-sweep-orig/matrices_hstacked \
#   --hparams_path hparams/llama31 \
#   --result_prefix cross-eval-orig-llama31

# GPTJ, extended dataset
# uv run python structrels/cross_eval.py \
#   --model gptj \
#   --data_path data/extended \
#   --num_samples 50 \
#   --decoder_folder results/gptj-sweep-extended/matrices_hstacked \
#   --hparams_path hparams/gptj \
#   --result_prefix cross-eval-extended-gptj

# NEOX, extended dataset
# uv run python structrels/cross_eval.py \
#   --model neox \
#   --data_path data/extended \
#   --num_samples 50 \
#   --decoder_folder results/neox-sweep-extended/matrices_hstacked \
#   --hparams_path hparams/neox \
#   --result_prefix cross-eval-extended-neox

# LLAMA31, extended dataset
# uv run python structrels/cross_eval.py \
#   --model llama31 \
#   --data_path data/extended \
#   --num_samples 50 \
#   --decoder_folder results/llama31-sweep-extended/matrices_hstacked \
#   --hparams_path hparams/llama31 \
#   --result_prefix cross-eval-extended-llama31

#!/bin/sh

# GPTJ, 50x50x50 block tensor network, extended dataset, all train split
uv run --env-file ./.env python structrels/main.py \
  --mt gptj \
  --model block \
  --outer_dim 4096 \
  --with_nn False \
  --inner_dim_r 50 \
  --inner_dim_s 50 \
  --inner_dim_o 50 \
  --inn_x 1 \
  --inn_y 1 \
  --inn_z 1 \
  --lr 0.001 \
  --optim_type SGD \
  --num_iters 9000 \
  --batch_size 16 \
  --exp_name model_block_comp \
  --run_id model \
  --seed 1 \
  --tag compression_extended \
  --folder_path data/extended \
  --split_by all_train \
  --relation_names_file data/extended/relations.txt \
  --hparams_path ./hparams/gptj/ \
  --decoders_path ./results/gptj-sweep-extended/matrices_hstacked/ \
  --log_freq 500

# GPTJ, 50x50x50 block tensor network, orig dataset, all train split
# uv run --env-file ./.env python structrels/main.py \
#   --mt gptj \
#   --model block \
#   --outer_dim 4096 \
#   --with_nn False \
#   --inner_dim_r 50 \
#   --inner_dim_s 50 \
#   --inner_dim_o 50 \
#   --inn_x 1 \
#   --inn_y 1 \
#   --inn_z 1 \
#   --lr 0.001 \
#   --optim_type SGD \
#   --num_iters 9000 \
#   --batch_size 16 \
#   --exp_name model_block_comp \
#   --run_id model \
#   --seed 1 \
#   --tag compression_orig \
#   --folder_path data/orig \
#   --split_by all_train \
#   --relation_names_file data/orig/relations.txt \
#   --hparams_path ./hparams/gptj/ \
#   --decoders_path ./results/gptj-sweep-orig/matrices_hstacked/ \
#   --log_freq 500

#!/bin/sh

# GPTJ, 50x50x50 block tensor network, math dataset, relation-wise split
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
  --exp_name model_block_gen_math \
  --run_id model \
  --seed 1 \
  --tag gen_math \
  --folder_path data/math \
  --split_by relations \
  --relation_names_file data/math/relations.txt \
  --hparams_path ./hparams/gptj/ \
  --decoders_path ./results/gptj-sweep-math/matrices_hstacked/ \
  --log_freq 500

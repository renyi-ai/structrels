#!/bin/sh

# GPTJ, 50x50x50, 50, 50, 50 triangle tensor network, extended dataset
uv run --env-file ./.env python structrels/main.py \
  --mt gptj \
  --model block \
  --outer_dim 4096 \
  --with_nn False \
  --inner_dim_r 50 \
  --inner_dim_s 50 \
  --inner_dim_o 50 \
  --inn_x 50 \
  --inn_y 50 \
  --inn_z 50 \
  --lr 0.001 \
  --optim_type SGD \
  --num_iters 9000 \
  --batch_size 16 \
  --exp_name model_triangle_gen_extended \
  --run_id model \
  --seed 1 \
  --tag gen_triangle_extended \
  --folder_path data/extended \
  --split_by relations \
  --relation_names_file data/extended/relations.txt \
  --hparams_path ./hparams/gptj/ \
  --decoders_path ./results/gptj-sweep-extended/matrices_hstacked/ \
  --log_freq 500

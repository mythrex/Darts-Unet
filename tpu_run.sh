#!/bin/bash
cd ~/Darts-Unet

python cnn/train_tpu_estimator.py \
--max_steps=5000 \
--mode='train' \
--train_batch_size=16 \
--eval_batch_size=16 \
--tpu='unet-darts' \
--zone='us-central1-f' \
--project='isro-nas' \
--model_dir='gs://unet-darts/train-search-ckpts' \
--data="gs://unet-darts/datasets/train-search" \
--num_layers=5 \
--num_shards=8 \
--steps_per_eval=500 \
--save_checkpoints_steps=500 \
--iterations_per_loop=500 \
--num_train_examples=1728 \
--use_host_call=True \
--learning_rate=0.025 \
--learning_rate_min=1e-10 \
--arch_learning_rate=3e-4 \
--seed=7

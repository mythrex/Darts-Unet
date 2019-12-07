#!/bin/bash

python cnn/train_tpu_estimator.py \
--max_steps=10000 \
--mode='train' \
--train_batch_size=16 \
--eval_batch_size=16 \
--tpu='unet-darts' \
--zone='us-central1-f' \
--project='isro-nas' \
--model_dir='gs://unet-darts/train-search-ckpts' \
--data="gs://unet-darts/datasets/train-search" \
--num_layers=3 \
--num_shards=8 \
--steps_per_eval=500 \
--save_checkpoints_steps=500 \
--iterations_per_loop=500 \
--num_train_examples=1728 \
--use_host_call=True

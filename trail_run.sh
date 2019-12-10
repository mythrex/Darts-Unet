#!/bin/bash
cd ~/Darts-Unet
python cnn/train_estimator.py \
--max_steps=100 \
--mode='train' \
--batch_size=2 \
--model_dir='gs://unet-darts/train-search-ckpts' \
--data="gs://unet-darts/datasets/train-search" \
--num_layers=3 \
--throttle_secs=100 \
--save_checkpoints_steps=100 \
--learning_rate=0.001

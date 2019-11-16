#!/bin/bash

python cnn/train_tpu_estimator.py \
--max_steps=1000 \
--mode='train_eval' \
--batch_size=2 \
--steps=1 \
--throttle_secs=60 \
--tpu='unet-darts' \
--zone='us-central1-f' \
--project='isro-nas' \
--model_dir='gs://unet-darts/model_ckpt' \
--data="gs://unet-darts/datasets"
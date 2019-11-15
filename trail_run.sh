#!/bin/bash

python cnn/train_estimator.py \
--max_steps=1000 \
--mode='train' \
--batch_size=2 \
--steps=1 \
--throttle_secs=60
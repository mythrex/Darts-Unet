
#!/bin/bash

python cnn/train_tpu_estimator.py \
--max_steps=10000 \
--mode='train_eval' \
--train_batch_size=8 \
--throttle_secs=120 \
--start_delay_secs=60 \
--tpu='unet-darts' \
--zone='us-central1-f' \
--project='isro-nas' \
--model_dir='gs://unet-darts/train-search-ckpts' \
--data="gs://unet-darts/datasets/train-search" \
--iterations_per_loop=100 \
--num_layers=3 \
--num_shards=8

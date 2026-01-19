#!/bin/bash

# Run the TimeKAN model for classification task

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path your files path \
  --model_id classification_example \
  --model TimeKAN \
  --data classification \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 7 \
  --d_model 64 \
  --d_ff 128 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 5 \
  --patience 10 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --begin_order 1 \
  --num_classes 7 \
  --use_gpu True \
  --gpu 0 \
  --loss CrossEntropy \
  --num_workers 0
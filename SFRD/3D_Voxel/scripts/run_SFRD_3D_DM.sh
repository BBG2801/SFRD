#!/bin/bash
set -e

cd ..
cuda_id=0
dst="ModelNet"
res=32
net="Conv3DNet"
ipc=1
sh_file="run_SFRD.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

batch_syn=0
dipc=0

lr_nf=1e-6
epochs_init=5000
lr_nf_init=5e-4

num_layers=4
layer_size=80
w0_initial=30
w0=40

use_relation_distill=1
lambda_rel=0.3
batch_real_rel=8
syn_decode_chunk=16

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DM.py \
  --dataset ${dst} --res ${res} \
  --model ${net} \
  --ipc ${ipc} \
  --sh_file ${sh_file} \
  --eval_mode ${eval_mode} \
  --data_path ${data_path} --save_path ${save_path} \
  --batch_syn ${batch_syn} \
  --dipc ${dipc} \
  --num_layers ${num_layers} \
  --layer_size ${layer_size} \
  --w0_initial ${w0_initial} \
  --w0 ${w0} \
  --lr_nf ${lr_nf} \
  --epochs_init ${epochs_init} \
  --lr_nf_init ${lr_nf_init} \
  --train_backbone 1 \
  --train_latent 1 \
  --init_batch_per_step 8 \
  --vis_threshold 0.5 \
  --use_relation_distill ${use_relation_distill} \
  --lambda_rel ${lambda_rel} \
  --batch_real_rel ${batch_real_rel} \
  --syn_decode_chunk ${syn_decode_chunk} \
  --FLAG "stable_init"

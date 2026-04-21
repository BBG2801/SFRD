#!/bin/bash
set -e

cd ..

cuda_id=0
dst="miniUCF101"
net="ConvNet3D"
ipc=5
sh_file="run_SFRD.sh"
eval_mode="SS"
data_path="/root/autodl-tmp/gitprojects/DDiF/DDiF/Video/distill_utils/data"
save_path="./results"

batch_syn=0
dipc=5
lr_nf=1e-5
frames=16

# SFRD shared-field + translation-only modulation
shared_mode="per_class"
shared_num_layers=6
shared_layer_size=60
shift_init=0.0
latent_std=0.01

# init / train lr
lr_nf_backbone=1e-5
lr_nf_shift=1e-4
lr_nf_init_backbone=1e-4
lr_nf_init_shift=0

# relation distillation
use_relation_distill=1
lambda_rel=0.3
batch_real_rel=8
syn_decode_chunk=16

CUDA_VISIBLE_DEVICES=${cuda_id} python distill_SFRD.py \
  --dataset ${dst} \
  --model ${net} \
  --ipc ${ipc} \
  --sh_file ${sh_file} \
  --eval_mode ${eval_mode} \
  --data_path ${data_path} \
  --save_path ${save_path} \
  --batch_syn ${batch_syn} \
  --dipc ${dipc} \
  --lr_nf ${lr_nf} \
  --frames ${frames} \
  --train_backbone \
  --train_latent \
  --lr_nf_backbone ${lr_nf_backbone} \
  --lr_nf_shift ${lr_nf_shift} \
  --lr_nf_init_backbone ${lr_nf_init_backbone} \
  --lr_nf_init_shift ${lr_nf_init_shift} \
  --shared_mode ${shared_mode} \
  --shared_num_layers ${shared_num_layers} \
  --shared_layer_size ${shared_layer_size} \
  --shift_init ${shift_init} \
  --latent_std ${latent_std} \
  --use_relation_distill ${use_relation_distill} \
  --lambda_rel ${lambda_rel} \
  --batch_real_rel ${batch_real_rel} \
  --syn_decode_chunk ${syn_decode_chunk} \
  --preload
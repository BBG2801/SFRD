#!/bin/bash
set -e

cd ..

cuda_id=0
dst="ImageNet"
subset="nette"
res=128
net="ConvNetD5"
ipc=1
eval_mode="S"
sh_file="run_SFRD.sh"

data_path="/root/autodl-tmp/gitprojects/linear-gradient-matching/data/datasets/imagenet"
save_path="./results"
buffer_path="/root/autodl-tmp/gitprojects/SFRD/buffers"

seed=0
FLAG=""
dipc=0

num_eval=5
Iteration=15000
batch_real=256
batch_train=256

batch_syn=60
micro_batch_syn=20
syn_steps=40
unroll_chunk_size=4

dim_in=2
dim_out=3
num_layers=4
layer_size=160
w0_initial=30
w0=40

shared_mode="global"
shared_num_layers=4
shared_layer_size=128
modulation_type="shift"
shift_init=0.0
latent_std=0.01

lr_nf=1e-6
lr_nf_init=1e-4
lr_nf_backbone=1e-6
lr_nf_shift=1e-4
lr_nf_init_backbone=1e-4
lr_nf_init_shift=5e-4

epochs_init=8000
init_instances_per_epoch=320
init_batch_per_step=80

train_backbone=True
train_latent=True

use_relation_distill=True
lambda_rel=0.3
batch_real_rel=8
syn_decode_chunk=32

dsa="True"
dsa_strategy="color_crop_cutout_flip_scale_rotate"
zca_flag=""
force_save_flag=""

CUDA_VISIBLE_DEVICES=${cuda_id} python main_TM.py \
  --dataset ${dst} \
  --subset ${subset} \
  --res ${res} \
  --model ${net} \
  --ipc ${ipc} \
  --sh_file ${sh_file} \
  --eval_mode ${eval_mode} \
  --num_eval ${num_eval} \
  --Iteration ${Iteration} \
  --data_path ${data_path} \
  --save_path ${save_path} \
  --buffer_path ${buffer_path} \
  --seed ${seed} \
  --FLAG "${FLAG}" \
  --batch_real ${batch_real} \
  --batch_train ${batch_train} \
  --batch_syn ${batch_syn} \
  --micro_batch_syn ${micro_batch_syn} \
  --syn_steps ${syn_steps} \
  --unroll_chunk_size ${unroll_chunk_size} \
  --truncate_unroll True \
  --empty_cache_every 1 \
  --debug_mem False \
  --dipc ${dipc} \
  --dim_in ${dim_in} \
  --dim_out ${dim_out} \
  --num_layers ${num_layers} \
  --layer_size ${layer_size} \
  --w0_initial ${w0_initial} \
  --w0 ${w0} \
  --lr_nf ${lr_nf} \
  --lr_nf_init ${lr_nf_init} \
  --lr_nf_backbone ${lr_nf_backbone} \
  --lr_nf_shift ${lr_nf_shift} \
  --lr_nf_init_backbone ${lr_nf_init_backbone} \
  --lr_nf_init_shift ${lr_nf_init_shift} \
  --epochs_init ${epochs_init} \
  --init_instances_per_epoch ${init_instances_per_epoch} \
  --init_batch_per_step ${init_batch_per_step} \
  --shared_mode ${shared_mode} \
  --shared_num_layers ${shared_num_layers} \
  --shared_layer_size ${shared_layer_size} \
  --modulation_type ${modulation_type} \
  --shift_init ${shift_init} \
  --latent_std ${latent_std} \
  --train_backbone ${train_backbone} \
  --train_latent ${train_latent} \
  --use_relation_distill ${use_relation_distill} \
  --lambda_rel ${lambda_rel} \
  --batch_real_rel ${batch_real_rel} \
  --syn_decode_chunk ${syn_decode_chunk} \
  --dsa ${dsa} \
  --dsa_strategy ${dsa_strategy} \
  ${zca_flag} \
  ${force_save_flag}

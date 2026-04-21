#!/bin/bash
set -e

cd ..
cuda_id=0
dst="ImageNet"
subset="nette"
res=128
net="ConvNetD5"
ipc=1
sh_file="run_SFRD_DC.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

batch_syn=0
dipc=0
seed=0
FLAG=""
empty_cache_every=1

lr_nf=5e-5
dim_in=2
dim_out=3
num_layers=3
layer_size=20
w0_initial=30
w0=40
shared_mode="global"
shared_num_layers=4
shared_layer_size=160
modulation_type="shift"
shift_init=0.0
latent_std=0.01

lr_nf_init=5e-4
lr_nf_backbone=${lr_nf}
lr_nf_shift=$(python - <<'PY'
print(5*5e-5)
PY
)
lr_nf_init_backbone=${lr_nf_init}
lr_nf_init_shift=2.5e-3
train_backbone=True
train_latent=True

epochs_init=5000
init_instances_per_epoch=-1
init_batch_per_step=32

use_relation_distill=True
lambda_rel=0.3
batch_real_rel=8
syn_decode_chunk=0

zca_flag=""
force_save_flag=""

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DC.py \
  --method SFRD \
  --objective DC \
  --dataset ${dst} --subset ${subset} --res ${res} \
  --model ${net} \
  --ipc ${ipc} \
  --sh_file ${sh_file} \
  --eval_mode ${eval_mode} \
  --data_path ${data_path} --save_path ${save_path} \
  --batch_syn ${batch_syn} \
  --dipc ${dipc} \
  --seed ${seed} \
  --FLAG "${FLAG}" \
  --empty_cache_every ${empty_cache_every} \
  --dim_in ${dim_in} \
  --dim_out ${dim_out} \
  --num_layers ${num_layers} \
  --layer_size ${layer_size} \
  --w0_initial ${w0_initial} \
  --w0 ${w0} \
  --lr_nf ${lr_nf} \
  --lr_nf_init ${lr_nf_init} \
  --epochs_init ${epochs_init} \
  --shared_mode ${shared_mode} \
  --shared_num_layers ${shared_num_layers} \
  --shared_layer_size ${shared_layer_size} \
  --modulation_type ${modulation_type} \
  --shift_init ${shift_init} \
  --latent_std ${latent_std} \
  --lr_nf_backbone ${lr_nf_backbone} \
  --lr_nf_shift ${lr_nf_shift} \
  --lr_nf_init_backbone ${lr_nf_init_backbone} \
  --lr_nf_init_shift ${lr_nf_init_shift} \
  --init_instances_per_epoch ${init_instances_per_epoch} \
  --init_batch_per_step ${init_batch_per_step} \
  --train_backbone ${train_backbone} \
  --train_latent ${train_latent} \
  --use_relation_distill ${use_relation_distill} \
  --lambda_rel ${lambda_rel} \
  --batch_real_rel ${batch_real_rel} \
  --syn_decode_chunk ${syn_decode_chunk} \
  ${zca_flag} \
  ${force_save_flag}

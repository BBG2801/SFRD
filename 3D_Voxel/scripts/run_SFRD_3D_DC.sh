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
lr_nf=1e-4

use_relation_distill=1
lambda_rel=0.3
batch_real_rel=8
syn_decode_chunk=16

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DC.py \
  --dataset ${dst} --res ${res} \
  --model ${net} \
  --ipc ${ipc} \
  --sh_file ${sh_file} \
  --eval_mode ${eval_mode} \
  --data_path ${data_path} --save_path ${save_path} \
  --batch_syn ${batch_syn} \
  --dipc ${dipc} \
  --lr_nf ${lr_nf} \
  --use_relation_distill ${use_relation_distill} \
  --lambda_rel ${lambda_rel} \
  --batch_real_rel ${batch_real_rel} \
  --syn_decode_chunk ${syn_decode_chunk}

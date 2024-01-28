#!/bin/bash -e
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }

set -e -o pipefail

export COLLECT_ROOT=$HOME/data/collections
dataset=msmarco_v1
model_type=parade_lateir_transf
model_conf=model_conf/config_long_parade_lateir_transf_pretr_L6.json
gpu_qty=`nvidia-smi -L|wc -l`
batch_sync_qty=4
train_data=cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17/text_raw
epoch_qty=3

./train_nn/train_model.sh  \ 
    -add_exper_subdir $model_conf \
    -epoch_qty $epoch_qty \
    -seed $seed \
    -batch_sync_qty $batch_sync_qty \
    -device_qty $gpu_qty \
    $dataset \
    $train_data \
    $model_type \
    -json_conf $model_conf 2>&1|tee log.train_model.$model_type.$model_conf


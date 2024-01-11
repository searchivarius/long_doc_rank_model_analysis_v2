#!/bin/bash -e
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }
export COLLECT_ROOT=$HOME/data/collections
dataset=msmarco_v1
model_type=vanilla_bert_stand
model_conf=model_conf/config_long_jina_base_v2.json
gpu_qty=`nvidia-smi -L|wc -l`
batch_sync_qty=4
epoch_qty=2
train_data=cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17/text_raw
./train_nn/train_model.sh  \
    -epoch_qty $epoch_qty \
    -add_exper_subdir $model_conf \
    -seed $seed \
    -batch_sync_qty $batch_sync_qty \
    -device_qty $gpu_qty \
    $dataset \
    $train_data \
    $model_type \
    -json_conf $model_conf 2>&1|tee log.train_model.$model_type.$model_conf




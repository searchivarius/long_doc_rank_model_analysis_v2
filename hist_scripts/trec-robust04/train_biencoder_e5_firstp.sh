#!/bin/bash
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }
fold=$2
[ "$fold" != "" ] || { echo "Specify fold " ; exit 1 ; }
field=$3
[ "$field" = "title" -o "$field" = "description" ] || { echo "Specify field: title or description " ; exit 1 ; }

set -e -o pipefail

export COLLECT_ROOT=$HOME/data/collections
dataset=trec-robust04
model_type=biencoder_e5
model_subdir=firstP/dwzhu/e5-base-4k
model_conf=config_e5.json
train_data=cedr_robust04_${field}_1000_1000_0_100_0_s0_fold${fold}_train/text_raw/
init_model=$COLLECT_ROOT/msmarco_v1/derived_data/ir_models/$model_type/$model_subdir/$seed/model.best

./train_nn/train_model.sh  \
     -epoch_repeat_qty 0 -batches_per_train_epoch 0 \
     -init_model $init_model \
     -add_exper_subdir $field/$model_subdir/fold${fold} \
     -seed $seed \
     $dataset \
     $train_data \
     $model_type \
     -json_conf model_conf/$model_conf 2>&1|tee log.train_model.$model_type.$model_conf

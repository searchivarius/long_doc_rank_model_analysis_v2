#!/bin/bash
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }
model_type=$2
[ "$model_type" != "" ] || { echo "Specify model type " ; exit 1 ; }
model_conf=$3
[ "$model_conf" != "" ] || { echo "Specify model configuration file (wihout model_conf prefix) " ; exit 1 ; }

set -e -o pipefail

export COLLECT_ROOT=$HOME/data/collections
dataset=msmarco_synthetic_longdoc
train_data=cedr_train_qty_50000_100_100_5_s0_bitext/text_raw
init_model=$COLLECT_ROOT/msmarco_v1/derived_data/ir_models/$model_type/model_conf/$model_conf/$seed/model.best
epoch_qty=1

cd $HOME/src/FlexNeuART/scripts 

./train_nn/train_model.sh  \
     -init_model $init_model \
     -add_exper_subdir $model_conf \
     -epoch_qty $epoch_qty \
     -seed $seed \
     $dataset \
     $train_data \
     $model_type \
     -json_conf model_conf/$model_conf 2>&1|tee log.train_model.$model_type.$model_conf

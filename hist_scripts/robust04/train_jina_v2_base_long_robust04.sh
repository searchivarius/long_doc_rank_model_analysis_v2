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
model_type=vanilla_bert_stand
model_conf=config_long_jina_base_v2.json
train_data=cedr_robust04_${field}_1000_1000_0_100_0_s0_fold${fold}_train/text_raw/
init_model=$COLLECT_ROOT/msmarco_v1/derived_data/ir_models/$model_type/model_conf/$model_conf/$seed/model.best
epoch_repeat_qty=100

cd $HOME/src/FlexNeuART/scripts 

./train_nn/train_model.sh  \
     -epoch_repeat_qty $epoch_repeat_qty \
     -init_model $init_model \
     -add_exper_subdir $field/$model_conf/fold${fold} \
     -seed $seed \
     $dataset \
     $train_data \
     $model_type \
     -json_conf model_conf/$model_conf 2>&1|tee log.train_model.$model_type.$model_conf

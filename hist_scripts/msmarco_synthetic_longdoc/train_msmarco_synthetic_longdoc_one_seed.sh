#!/bin/bash
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }

set -e -o pipefail

export COLLECT_ROOT=$HOME/data/collections
DATASET=msmarco_synthetic_longdoc
COLLECT_DIR=$COLLECT_ROOT/$DATASET
[ ! -d $COLLECT_DIR ] && { echo "Missing dir: $COLLECT_DIR" ; exit 1 ; }
MODEL_ROOT_DIR=$COLLECT_ROOT/msmarco_v1/derived_data/ir_models/

read_link_path=$(readlink -f "$0")
abs_path=$(dirname "$read_link_path")
echo $abs_path

echo "Searching in $MODEL_ROOT_DIR"
for train_stat in `find $MODEL_ROOT_DIR -name train_stat.json|fgrep ".json/$seed/"` ; do
    model_dir=`dirname $train_stat`
    # Hacky extraction of model type and config from the path
    model_type=`echo $model_dir|sed 's/^.*ir_models[/]*//'| sed 's/[/].*$//'`
    model_conf=`echo $model_dir|sed 's/^.*model_conf[/]*//'| sed 's/[/].*$//'`
    echo $model_dir
    echo $model_type $model_conf

    $abs_path/train_msmarco_synthetic_longdoc_one_model_one_seed.sh $seed $model_type $model_conf
done

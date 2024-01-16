#!/bin/bash -e
set -e -o pipefail
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }
train_script=$2
[ "$train_script" != "" ] || { echo "Specify training script " ; exit 1 ; }
fold_qty=5

read_link_path=$(readlink -f "$0")
abs_path=$(dirname "$read_link_path")
echo $abs_path

for field in title description ; do
    for ((fold=1; fold <= $fold_qty; fold++)) ; do
        echo $field $fold_qty $train_script
        echo "=================================="
        $abs_path/$train_script $seed $fold $field
    done
done

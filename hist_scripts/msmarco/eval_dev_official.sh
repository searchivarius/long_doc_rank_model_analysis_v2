#!/bin/bash -e
seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }

export COLLECT_ROOT=$HOME/data/collections
DATASET=msmarco_v1
COLLECT_DIR=$COLLECT_ROOT/$DATASET
[ ! -d $COLLECT_DIR ] && { echo "Missing dir: $COLLECT_DIR" ; exit 1 ; }
DEVICE_NAME=cuda:0
EVAL_RERUN_STAT=eval_rerank.json
BATCH_SIZE=16
# All models come from the MS MARCO catalog
MODEL_ROOT_DIR=$COLLECT_DIR/derived_data/ir_models

for part in dev_official ; do
  RUN_FILE=$COLLECT_DIR/derived_data/trec_runs_cached/$DATASET/$part/run_100.bz2

  if [ "$part" = "dev_official" ] ; then
    METRIC=recip_rank
  else
    METRIC=ndcg@10
  fi

  echo "Searching in $MODEL_ROOT_DIR"

  for train_stat in `find $MODEL_ROOT_DIR -name train_stat.json|fgrep ".json/$seed/"` ; do
    model_dir=`dirname $train_stat`
    dst_dir=$model_dir/$part
    if [ ! -d "$dst_dir" ] ; then
        echo "Making dir: $dst_dir"
        mkdir $dst_dir
    fi

    echo $part $METRIC $train_stat

    out_summary_file=$model_dir/$part/$EVAL_RERUN_STAT
    out_rerank_file=$model_dir/$part/run_rerank.txt
    if [ -f "$out_summary_file" -a -f "$out_rerank_file" ] ; then 
        echo "Skipping!"
        continue
    else
        echo "Running eval!"
        echo $out_rerank_file
        echo $out_summary_file
    fi

    ./train_nn/eval_model.py \
      --amp \
      --batch_size $BATCH_SIZE \
      --eval_metric $METRIC \
      --device_name $DEVICE_NAME \
      --init_model $model_dir/model.best \
      --collect_dir $COLLECT_DIR  \
      --run_orig $RUN_FILE \
      --run_rerank $out_rerank_file \
      --summary_json $out_summary_file  \
      --qrels $COLLECT_DIR/input_data/$part/qrels.txt \
      --query_file $COLLECT_DIR/input_data/$part/QuestionFields.jsonl  \
      --fwd_index_subdir forward_index \
      --index_field text_raw
  
  done
done

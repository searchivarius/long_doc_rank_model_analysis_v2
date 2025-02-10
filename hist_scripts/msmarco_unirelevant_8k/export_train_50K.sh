#!/bin/bash -e
export COLLECT_ROOT=$HOME/data/collections
dataset=msmarco_unirelevant_8k

hardNegQty=0
# change the seed from the previous run
seed=0

# We are gonna run it for one epoch only
sample_neg_qty=5

N=100

cand_prov=trec_runs
cand_prov_uri=derived_data/trec_runs_cached/$dataset/all/run_${N}.bz2

cand_train_qty=${N}
#candTrain4PosQty=1000

train_qty=50000
max_num_query_train_opt=" -max_num_query_train $train_qty"
#max_num_query_test_opt=" -max_num_query_test 1000"

test_part=dev_official_sample1K

cand_test_qty=${N}
thread_qty=4

field_name=text_raw 
train_part=bitext

out_subdir=cedr_train_qty_${train_qty}_${cand_train_qty}_${cand_test_qty}_${sample_neg_qty}_s${seed}_$train_part/$field_name

# Usage: <collection> <name of the index field> <train subdir, e.g., train_fusion> <test subdir, e.g., dev1>
./export_train/export_cedr.sh \
      $dataset \
      $field_name \
      $train_part \
      $test_part \
     -max_doc_whitespace_qty 1536 \
      -cand_prov $cand_prov \
      -cand_prov_uri $cand_prov_uri \
      $max_num_query_train_opt \
      $max_num_query_test_opt \
      -thread_qty $thread_qty \
      -out_subdir $out_subdir \
      \
      -sample_med_neg_qty $sample_neg_qty \
       \
      -cand_train_qty $cand_train_qty \
      -cand_test_qty $cand_test_qty  2>&1 | tee log.export_train.$col


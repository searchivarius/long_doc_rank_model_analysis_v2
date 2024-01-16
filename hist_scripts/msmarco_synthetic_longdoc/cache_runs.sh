#!/bin/bash -e
export COLLECT_ROOT="$HOME/data/collections"
export collect=msmarco_synthetic_longdoc
RUN_SUBDIR=bm25_lucene
collect_dir=$COLLECT_ROOT/$collect
N=100

set -e -o pipefail

for part in bitext dev_official_sample1K ; do
  add_opt=""
  if [ "$part" = "bitext" ] ; then
    add_opt=" -skip_eval "
  fi

  ./exper/run_experiments.sh $collect exper_desc/${RUN_SUBDIR}_test.json -test_part $part -no_separate_shell -test_cand_qty_list ${N} $add_opt

  dst_dir="$collect_dir/derived_data/trec_runs_cached/$collect/$part"
  mkdir -p $dst_dir

  cp $collect_dir/results/$part/final_exper/$RUN_SUBDIR/trec_runs/run_${N}.bz2 $dst_dir

done
part="all"
dst_dir_all="$collect_dir/derived_data/trec_runs_cached/$collect/$part"
mkdir -p $dst_dir_all
echo > $dst_dir_all/run_${N}
for N in 100 ; do
for part in bitext dev_official_sample1K ; do
  dst_dir="$collect_dir/derived_data/trec_runs_cached/$collect/$part"
  bzcat $dst_dir/run_${N}.bz2 >> $dst_dir_all/run_${N} 
done
done

bzip2 $dst_dir_all/run_${N} 

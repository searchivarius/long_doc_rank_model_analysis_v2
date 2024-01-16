#!/bin/bash -e 
COL=trec-robust04
INPUT_DIR=/disk3/collections/$COL/input_data

overlap() {
  fn1="$1"
  fn2="$2"
  t1=`mktemp`
  t2=`mktemp`
  sort $fn1 > $t1
  sort $fn2 > $t2
  comm -12 $t1 $t2 
  rm -f $t1 $t2
}

TMP_QUERY=/tmp/merge_queries
TMP_QRELS=/tmp/merge_qrels
for ((fld=1;fld <= 5;fld++)) ; do
  dst_dir=$INPUT_DIR/fold${fld}_train
  dst_queries=$dst_dir/QuestionFields.jsonl
  dst_qrels=$dst_dir/qrels.txt
  echo "Generating fold $fld destination: $dst_dir"
  echo -n "" > $TMP_QUERY
  echo -n "" > $TMP_QRELS
  for ((fc=1;fc <= 5;fc++)) ; do
      if [ "$fc" != "$fld" ] ; then
          src_dir=fold${fc}_test
          echo "Copying from $src_dir"
          src_queries=$INPUT_DIR/$src_dir/QuestionFields.jsonl
          src_qrels=$INPUT_DIR/$src_dir/qrels.txt
          cat $src_queries >> $TMP_QUERY  
          cat $src_qrels >> $TMP_QRELS  
      fi
  done
  if [ ! -d $dst_dir ] ; then
      mkdir -p $dst_dir
  fi
  cat $TMP_QUERY > $dst_queries 
  cat $TMP_QRELS > $dst_qrels
  for ((fc=1;fc <= 5;fc++)) ; do
      src_queries=$INPUT_DIR/fold${fc}_test/QuestionFields.jsonl
      src_qrels=$INPUT_DIR/fold${fc}_test/qrels.txt
      if [ "$fc" != "$fld" ] ; then
          echo "Overlap is expected:"
          overlap $src_queries $dst_queries | wc -l
          overlap $src_qrels $dst_qrels | wc -l
          echo "============="
      else
          echo "Overlap is *NOOOT* expected:"
          overlap $src_queries $dst_queries | wc -l
          overlap $src_qrels $dst_qrels | wc -l
          echo "============="
      fi
  done
done


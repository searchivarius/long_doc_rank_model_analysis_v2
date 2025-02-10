#!/bin/bash
export COLLECT_ROOT=$HOME/data/collections
dataset=msmarco_unirelevant_8k
./index/create_fwd_index.sh $dataset text_raw:textRaw  -clean
./index/create_lucene_index.sh  $dataset -index_field doc2query_text_fusion 

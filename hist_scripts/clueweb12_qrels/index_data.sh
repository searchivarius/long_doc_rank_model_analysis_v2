#!/bin/bash
export COLLECT_ROOT=$HOME/data/collections
dataset=clueweb12_qrels
./index/create_fwd_index.sh $dataset text_raw:textRaw  -clean

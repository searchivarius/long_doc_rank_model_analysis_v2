#!/bin/bash -e
sudo yum install maven -y

mkdir -p $HOME/data/collections
export COLLECT_ROOT=$HOME/data/collections
DATASET=msmarco_v1
mkdir -p $COLLECT_ROOT/$DATASET/derived_data
mkdir -p $COLLECT_ROOT/$DATASET/model_conf
mkdir -p $COLLECT_ROOT/$DATASET/hist_scripts

mkdir $HOME/src
cd $HOME/src
git clone https://github.com/searchivarius/long_doc_rank_model_analysis.git
cp long_doc_rank_model_analysis/model_conf/main/* $COLLECT_ROOT/$DATASET/model_conf
cp -r long_doc_rank_model_analysis/trec_runs_cached/ $COLLECT_ROOT/$DATASET/derived_data
git clone https://github.com/oaqa/FlexNeuART.git --branch torch_1.13.1_2023
cd FlexNeuART
pip install .
cd $HOME


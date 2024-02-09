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
git clone https://github.com/searchivarius/long_doc_rank_model_analysis_v2.git
cp long_doc_rank_model_analysis/model_conf/main/* $COLLECT_ROOT/$DATASET/model_conf
cp -r long_doc_rank_model_analysis/trec_runs_cached/ $COLLECT_ROOT/$DATASET/derived_data
git clone https://github.com/oaqa/FlexNeuART.git --branch torch_1.13.1_2023
cd FlexNeuART
pip install .
cd $HOME


cd $COLLECT_ROOT/$DATASET/derived_data
wget https://file.io/yVisDnLlS2Zs
mv yVisDnLlS2Zs cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2  
tar jxvf cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2

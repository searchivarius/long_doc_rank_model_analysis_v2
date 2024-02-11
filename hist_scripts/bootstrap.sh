#!/bin/bash -e
core_scripts_dir=1
[ ! -z "$core_scripts_dir" ] || { echo "Specify the directory to install core scripts" ; exit 1 ; }

sudo yum install maven -y

pip install flexneuart
flexneuart_install_extra.sh $core_scripts_dir 0

exper_repo_dst_dir=$core_scripts_dir/scripts/this_exper_repo

mkdir -p $exper_repo_dst_dir
cd $exper_repo_dst_dir

wget ...

mkdir -p $HOME/data/collections
export COLLECT_ROOT=$HOME/data/collections
DATASET=msmarco_v1
mkdir -p $COLLECT_ROOT/$DATASET/derived_data
mkdir -p $COLLECT_ROOT/$DATASET/model_conf
cd $COLLECT_ROOT/$DATASET/derived_data

DATASET=msmarco_synthetic_longdoc
cd $COLLECT_ROOT/$DATASET/derived_data
wget https://file.io/yVisDnLlS2Zs
mv yVisDnLlS2Zs cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2  
tar jxvf cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2

mkdir -p $COLLECT_ROOT/$DATASET/derived_data
mkdir -p $COLLECT_ROOT/$DATASET/model_conf
cp $exper_repo_dst_dir/model_conf/main/* $COLLECT_ROOT/$DATASET/model_conf

echo "All is done: cd to $core_scripts_dir before running any training scripts from $exper_repo_dst_dir/hist_scripts!"


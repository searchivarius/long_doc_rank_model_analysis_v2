#!/bin/bash -e
core_scripts_dir=$1
[ ! -z "$core_scripts_dir" ] || { echo "Specify the directory to install core scripts" ; exit 1 ; }
sudo yum install maven -y
pip install flexneuart
flexneuart_install_extra.sh $core_scripts_dir 0

exper_repo_dst_dir=$core_scripts_dir/this_exper_repo
mkdir -p $exper_repo_dst_dir
cd $exper_repo_dst_dir
wget https://file.io/CroBcsvkzFjq
mv CroBcsvkzFjq model_conf.tar.gz
tar zxvf model_conf.tar.gz

mkdir -p $HOME/data/collections
export COLLECT_ROOT=$HOME/data/collections
DATASET=msmarco_v1
mkdir -p $COLLECT_ROOT/$DATASET/derived_data
mkdir -p $COLLECT_ROOT/$DATASET/model_conf
cp $exper_repo_dst_dir/model_conf/main/* $COLLECT_ROOT/$DATASET/model_conf
cd $COLLECT_ROOT/$DATASET/derived_data
wget https://file.io/yVisDnLlS2Zs
mv yVisDnLlS2Zs cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2  
tar jxvf cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17.tar.bz2

DATASET=msmarco_synthetic_longdoc
cd $COLLECT_ROOT
wget https://file.io/Grh3kSiJn9Hq
mv Grh3kSiJn9Hq msmarco_synthetic_longdoc_2024-01-23.tar.bz2
tar jxvf  msmarco_synthetic_longdoc_2024-01-23.tar.bz2
mkdir -p $COLLECT_ROOT/$DATASET/derived_data
cd $COLLECT_ROOT/$DATASET/
ln -s $COLLECT_ROOT/msmarco_v1/model_conf
mkdir -p $COLLECT_ROOT/$DATASET/model_conf
cd $COLLECT_ROOT/$DATASET/derived_data

wget https://file.io/h2LUjDe3Y75l
mv h2LUjDe3Y75l cedr_mmarco_far_relevant_train_qty_50000_100_100_5_s0_bitext.tar.bz2
tar jxvf cedr_mmarco_far_relevant_train_qty_50000_100_100_5_s0_bitext.tar.bz2

echo "All is done: cd to $core_scripts_dir before running any training scripts from $exper_repo_dst_dir/hist_scripts!"


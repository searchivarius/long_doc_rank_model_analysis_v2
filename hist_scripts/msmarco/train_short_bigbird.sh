seed=$1
[ "$seed" != "" ] || { echo "Specify seed " ; exit 1 ; }
export COLLECT_ROOT=/home/ec2-user/data/collections
dataset=msmarco_v1
model_type=mosaic_bert
model_conf=model_conf/config_short_mosaic_base.json
gpu_qty=`nvidia-smi -L|wc -l`
batch_sync_qty=4
epoch_qty=2
export TOKENIZERS_PARALLELISM=False
train_data=cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17/text_raw
#    -batches_per_train_epoch 500 -max_query_val 250 \
./train_nn/train_model.sh  \
    -master_port 10100 \
    -epoch_qty $epoch_qty \
    -add_exper_subdir $model_conf \
    -seed $seed \
    -batch_sync_qty $batch_sync_qty \
    -device_qty $gpu_qty \
    $dataset \
    $train_data \
    $model_type \
    -json_conf $model_conf 2>&1|tee log.train_model.$model_type.$model_conf



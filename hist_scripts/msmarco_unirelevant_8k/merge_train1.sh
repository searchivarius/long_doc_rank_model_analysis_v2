#!/bin/bash -e


/export_train/mix_train_data.py \
  --input_dirs \
     /disk3/collections/msmarco_unirelevant_8k/derived_data/cedr_train_qty_30000_100_100_5_s0_bitext/text_raw/ \
     /disk3/collections_cleaned/msmarco_doc_v1/derived_data/cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17/text_raw/ \
  --output_dir \ 
     /disk3/collections_cleaned/msmarco_doc_v1/derived_data/cedr_mcds_100_50_0_5_0_s0_bitext_2021-11-17_unirelevant_8k_30K/text_raw/ \
  --seed 0 --prob 1 1


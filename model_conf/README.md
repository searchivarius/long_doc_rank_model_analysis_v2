# Model configurations

Below is the list of main model configuration files, whose evaluation results are present in the paper:

| model name                     | model code                  | configuration file                                 |
|--------------------------------|-----------------------------|----------------------------------------------------|
| AvgP                           | vanilla_bert                | config_long.json                                   |
| FirstP (BERT)                  | vanilla_bert                | config_short.json                                  |
| FirstP (Longformer)            | longformer                  | config_short_longformer.json                       |
| FirstP (ELECTRA)               | vanilla_bert                | config_short_electra.json                          |
| FirstP (DeBERTA)               | vanilla_bert                | config_short_deberta.json                          |
|                                |                             |                                                    |
| MaxP                           | bert_maxp                   | config_long.json                                   |
| SumP                           | bert_sump                   | config_long.json                                   |
|                                |                             |                                                    |
| CEDR-DRMM                      | cedr_drmm                   | config_long_drmm.json                              |
| CEDR-KNRM                      | cedr_knrm                   | config_long_knrm.json                              |
| CEDR-PACRR                     | cedr_pacrr                  | config_long_cedr_pacrr.json                        |
|                                |                             |                                                    |
| PARADE Attn                    | parade_attn                 | config_long_parade.json                            |
| PARADE Attn (ELECTRA)          | parade_attn                 | config_long_parade_electra.json                    |
| PARADE Avg                     | parade_avg                  | config_long_parade.json                            |
| PARADE Max                     | parade_max                  | config_long_parade.json                            |
|                                |                             |                                                    |
| PARADE Transf-RAND-L2          | parade_transf_rand          | config_long_parade_rand.json                       |
| PARADE Transf-RAND-L2          | parade_transf_rand          | config_long_parade_rand_electra.json               |
| PARADE Transf-PRETR-L6         | parade_transf_pretr         | config_long_parade_pretr_L6.json                   |
| PARADE-Transf-Pretr-LATEIR-L6  | parade_lateir_transf        | config_long_parade_lateir_transf_pretr_L6.json     |
|                                |                             |                                                    |
| Big-BIRD                       | vanilla_bert                | config_long_bigbird.json                           |
| Longformer                     | longformer                  | config_long_longformer.json                        |
| MOSAIC BERT                    | mosaic_bert                 | config_long_mosaic_base.json                       |
| JINA BERT                      | vanilla_bert_stand          | config_long_jina_base_v2.json                      |

We implemented and tried to test addditional models, but we experienced issues:
1. LongT5 models have been convering very slowly. After 10 epochs their accuracy was still about 10% below other models.
2. Roformer models worked only when input was truncated to 512 characters. Otherwise, they crashed with CUDA errors. 
For completeness, we list respective configuration files as well:

| model name                     | model code                  | configuration file                                 |
|--------------------------------|-----------------------------|----------------------------------------------------|
| Roformer                       | vanilla_bert_stand          | config_long_roformer.json                          |
| T5 encoder-only                | t5_enc                      | config_long_t5_base.json                           |
| T5 encoder-decoder             | t5_encdec                   | config_long_t5_base.json                           |



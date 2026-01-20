#!/usr/bin/env python 
import json
import os

from time import sleep
from tqdm.auto import tqdm
from time import time
from collections import defaultdict
import argparse
import os
from dotenv import load_dotenv
from rank_gpt_lib import sliding_windows

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.config import QUESTION_FILE_JSON
from flexneuart.retrieval.fwd_index import get_forward_index
from flexneuart.io.queries import read_queries_dict
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.text_proc.parse import Sentencizer

from cache_handler import CacheManagerJSON
from chunk_utils import extract_sentences_up_to_the_limit

from transformers import AutoTokenizer

RETRY_QTY=3
INIT_SLEEP_TIME=10
SLEEP_TIME_FACTOR=4

def main(args):
    config = json.load(open(args.config))
    model_name=config['model_name']

    if 'gpt' in model_name:
        api_key=os.environ['OPEN_AI_API_KEY']
    elif 'claude' in model_name:
        api_key=os.environ['CLAUDE_API_KEY']
    else:
        raise Exception(f'Unsupported model: {model_name}')


    collect_root=config['collect_root']
    dataset=config['dataset']
    dataset_result_dir=config['dataset_result_dir']
    dataset_dir=f'{collect_root}/{dataset}'
    window_size=config['window_size']
    step=config['step']
    rank_end=config['rank_end']
    query_part=config['query_part']
    trec_run_subdir=config.get('trec_run_subdir', None)
    prefix_config=config.get('prefix_config', None)
    

    if prefix_config is None:
        max_tok_qty_subdir=''
    else:
        max_tok_qty=prefix_config.get('max_tok_qty', None)
        max_tok_qty_subdir=f'_max_tok_qty_{max_tok_qty}'
        spacy_sentencizer=Sentencizer(prefix_config['spacy_model'])
        transformer_tokenizer = AutoTokenizer.from_pretrained(prefix_config['tokenizer_model'])
        heuristic_max_avg_tok_size=prefix_config['heuristic_max_avg_tok_size']

    query_field_name=config['query_field_name']
    doc_field_name=config['doc_field_name'] 

    print(f'Query field name: {query_field_name} document field name: {doc_field_name}')
    sample_qty = config.get('sample_qty', None)
    if sample_qty is None:
        sample_qty_subdir=''
    else:
        sample_qty_subdir=f'_sample_{sample_qty}'

    result_dir=f'{collect_root}/longp_results/{dataset_result_dir}/derived_data/ir_models/llm_ranker/model_conf/' + \
                f'{model_name}_{window_size}_{step}_{rank_end}{sample_qty_subdir}{max_tok_qty_subdir}/0/{query_part}'
    
    print(f'Writing results to {result_dir}')

    queries = read_queries_dict(f'{dataset_dir}/input_data/{query_part}/{QUESTION_FILE_JSON}')
    if trec_run_subdir is not None:
        runs = read_run_dict(f'{dataset_dir}/derived_data/trec_runs_cached/{trec_run_subdir}/run_{rank_end}.bz2')
    else:
        runs = read_run_dict(f'{dataset_dir}/derived_data/trec_runs_cached/{dataset}/{query_part}/run_{rank_end}.bz2')

    query_keys = list(sorted(queries.keys()))
    import numpy as np
    np.random.seed(0)
    np.random.shuffle(query_keys)

    doc_mngr=create_featextr_resource_manager(resource_root_dir=dataset_dir,
                                          fwd_index_dir='forward_index')
    doc_indx=get_forward_index(doc_mngr, doc_field_name)

    new_trec_run = defaultdict(dict)

    total_time = 0


    if sample_qty is None:
        sample_qty = len(query_keys)

    cache_dir = f'{result_dir}/cache'
    res_cache = CacheManagerJSON(cache_dir)


    print(f'Cache dir: {cache_dir}')
    print(f'Sample qty: {sample_qty}')

    proc_qty = 0
    for qid in tqdm(query_keys[0:sample_qty]):
        proc_qty += 1
        #if proc_qty > 10: break
        cache_rec = res_cache.get(qid)
        #print(res is not None)
        if not cache_rec:
            hits = []
            item = {
                'query': queries[qid][query_field_name],
                'hits': hits
            }
            for did in runs[qid]:
                doc_text = doc_indx.get_doc_raw(did)
                if prefix_config is not None:
                    doc_text = extract_sentences_up_to_the_limit(text=doc_text, 
                                                                 spacy_sentencizer=spacy_sentencizer,
                                                                 tok=transformer_tokenizer,
                                                                 max_tok_qty=max_tok_qty,
                                                                 heuristic_max_avg_tok_size=heuristic_max_avg_tok_size)
                hits.append(dict(content=doc_text, doc_id=did))
            t0 = time()
            error = None
            wait_time = INIT_SLEEP_TIME
            for att_id in range(RETRY_QTY):
                try:
                    new_item = sliding_windows(item, 
                                rank_start=0, rank_end=rank_end, 
                                window_size=window_size, step=step, 
                                model_name=model_name, 
                                api_key=api_key)
                    error = None
                    break
                except Exception as e:
                    error = e
                    print('Retrying after:', e, ' time to sleep', wait_time)
                    sleep(wait_time)
                    wait_time *= SLEEP_TIME_FACTOR

            if error is not None:
                print('Failing due to:', error) 
                raise error
            t1 = time()
            new_run = []
            for rec_id, rec in enumerate(new_item['hits']):
                assert rec['doc_id'] in runs[qid]
                new_run.append((rec['doc_id'], -rec_id))
            cache_rec = dict(query_id = qid, time=t1 - t0, run=new_run)
            res_cache.put(qid, cache_rec)

        total_time += cache_rec['time']
        assert qid == cache_rec['query_id']
        for doc_id, score in cache_rec['run']:
            new_trec_run[qid][doc_id] = score

    write_run_dict(new_trec_run, f'{result_dir}/run_rerank.txt')
    with open(f'{result_dir}/eval_rerank.json', 'w') as out_f:
        json.dump(dict(validation_time=total_time), out_f)


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    load_dotenv()
    configure_classpath()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()
    main(args)

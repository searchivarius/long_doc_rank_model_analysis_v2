#!/usr/bin/env python
import json
from tqdm.auto import tqdm
from time import time
from collections import defaultdict
import argparse
import numpy as np
import os


from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.config import QUESTION_FILE_JSON
from flexneuart.retrieval.fwd_index import get_forward_index
from flexneuart.io.queries import read_queries_dict
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.text_proc.parse import Sentencizer

from cache_handler import CacheManagerJSON

from transformers import AutoTokenizer

from llm_judge_utils import judge_query_doc_pair, extract_judgment
from chunk_utils import extract_n_chunks, combine_adjacent_chunks

CHUNK_SEP='\n'
MAX_RAND=0.1

def judge_query_doc_pair_chunk_doc(judge_gpt, qid, did, query_text, doc_text,
                                 spacy_sentencizer, transformer_tokenizer, chunk_config):
    doc_chunks = extract_n_chunks(doc_text, spacy_sentencizer, transformer_tokenizer,
                                    max_chunk_qty=chunk_config['max_chunk_qty'], max_chunk_size=chunk_config['max_chunk_size'],
                                    heuristic_max_avg_tok_size=chunk_config['heuristic_max_avg_tok_size'])
                                      
    doc_chunks_merged = combine_adjacent_chunks(doc_chunks, 
                                                span=chunk_config['chunk_merge_qty'], stride=chunk_config['chunk_merge_stride'],
                                                chunk_sep=CHUNK_SEP)

    chunk_scores = []

    for _, chunk_text in doc_chunks_merged:
        response = judge_query_doc_pair(judge_gpt, qid, did, query_text, chunk_text)
        chunk_scores.append(extract_judgment(response))

    return dict(max=max(chunk_scores), sum=sum(chunk_scores))

def main(args):
    np.random.seed(0)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = json.load(open(args.config))

    collect_root=config['collect_root']
    dataset=config['dataset']
    dataset_result_dir=config['dataset_result_dir']
    dataset_dir=f'{collect_root}/{dataset}'
    gpt_model=config['gpt_model']
    rank_end=config['rank_end']

    chunk_config = config['chunking']

    tokenizaton_config=config['tokenization']
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizaton_config['tokenizer_model'])
    spacy_sentencizer=Sentencizer(tokenizaton_config['spacy_model'])

    judge_gpt = GPTJudge(qrel="test_qrels", prompt_type="bing", engine=gpt_model)

    query_part=config['query_part']
    trec_run_subdir=config.get('trec_run_subdir', None)

    query_field_name=config['query_field_name']
    doc_field_name=config['doc_field_name'] 

    judge_type=config['judge_type']

    print(f'Query field name: {query_field_name} document field name: {doc_field_name}')
    sample_qty = config.get('sample_qty', None)
    if sample_qty is None:
        sample_qty_subdir=''
    else:
        sample_qty_subdir=f'_sample_{sample_qty}'

    result_dir=f'{collect_root}/longp_results/{dataset_result_dir}/derived_data/ir_models/llm_ranker_llm_judge_chunk_doc/model_conf/' + \
                f'{gpt_model}_{judge_type}{sample_qty_subdir}/0/{query_part}'
    
    print(f'Writing results to {result_dir}')
    
    queries = read_queries_dict(f'{dataset_dir}/input_data/{query_part}/{QUESTION_FILE_JSON}')
    if trec_run_subdir is not None:
        runs = read_run_dict(f'{dataset_dir}/derived_data/trec_runs_cached/{trec_run_subdir}/run_{rank_end}.bz2')
    else:
        runs = read_run_dict(f'{dataset_dir}/derived_data/trec_runs_cached/{dataset}/{query_part}/run_{rank_end}.bz2')

    query_keys = list(sorted(queries.keys()))

    np.random.seed(0)
    np.random.shuffle(query_keys)

    doc_mngr=create_featextr_resource_manager(resource_root_dir=dataset_dir,
                                          fwd_index_dir='forward_index')
    doc_indx=get_forward_index(doc_mngr, doc_field_name)

    new_trec_run_max = defaultdict(dict)
    new_trec_run_sum = defaultdict(dict)

    total_time = 0

    if sample_qty is None:
        sample_qty = len(query_keys)

    cache_dir = f'{result_dir}/cache'
    res_cache = CacheManagerJSON(cache_dir)


    print(f'Cache dir: {cache_dir}')
    print(f'Sample qty: {sample_qty}')

    np.random.seed(0)

    for qid in tqdm(query_keys[0:sample_qty]):
        cache_rec = res_cache.get(qid)
        #print(res is not None)
        if not cache_rec:
            query_text = queries[qid][query_field_name],
            run_max = []
            run_sum = []

            t0 = time()
            for did in runs[qid]:
                doc_text = doc_indx.get_doc_raw(did)
                
                score_dict = judge_query_doc_pair_chunk_doc(judge_gpt, qid, did, query_text, doc_text,
                                                     spacy_sentencizer, transformer_tokenizer, chunk_config)
                # We add a small < 1 random number to resolve/reandomize ties
                run_max.append((did, score_dict['max'] + np.random.uniform(0, MAX_RAND)))
                run_sum.append((did, score_dict['sum'] + np.random.uniform(0, MAX_RAND)))

            t1 = time()

            cache_rec = dict(query_id = qid, time=t1 - t0, run_max=run_max, run_sum=run_sum)
            res_cache.put(qid, cache_rec)

        total_time += cache_rec['time']
        assert qid == cache_rec['query_id']
        for doc_id, score in cache_rec['run_max']:
            new_trec_run_max[qid][doc_id] = score
        for doc_id, score in cache_rec['run_sum']:
            new_trec_run_sum[qid][doc_id] = score

    write_run_dict(new_trec_run_max, f'{result_dir}/run_max_rerank.txt')
    write_run_dict(new_trec_run_sum, f'{result_dir}/run_sum_rerank.txt')


    with open(f'{result_dir}/eval_rerank.json', 'w') as out_f:
        json.dump(dict(validation_time=total_time), out_f)


if __name__ == '__main__':
    load_dotenv()
    configure_classpath()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()
    main(args)

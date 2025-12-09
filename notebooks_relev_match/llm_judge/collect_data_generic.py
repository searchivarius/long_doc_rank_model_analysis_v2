#!/usr/bin/env python
import json
import os
import argparse

from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np

from flexneuart import configure_classpath
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.io.queries import read_queries_dict
from flexneuart.config import QUESTION_FILE_JSON, QREL_FILE
from flexneuart.text_proc.parse import Sentencizer

from cache_handler import CacheManagerJSON

from chunk_utils import extract_n_chunks, interleave_query_docs, combine_adjacent_chunks

from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv

from transformers import AutoTokenizer

import logging

CHUNK_SEP='\n'

from llm_judge_utils import judge_query_doc_pair

def proc_query_doc_pair(config, cache_mngr, doc_index, 
                        queries, qrels, 
                        spacy_sentencizer, transformer_tokenizer, judge_gpt,
                        qid, did):
    human_relev_grade_treshold = config['human_relev_grade_treshold']
    qdoc_key = f'qid_{qid}_did_{did}'

    rec = cache_mngr.get(qdoc_key)

    chunk_config = config['chunking']

    query_text = queries[qid][config['query_field_name']]
    doc_text = doc_index.getDocEntryTextRaw(did)
    assert doc_text is not None, f'No text for document ID: {did}'

    doc_chunks = extract_n_chunks(doc_text, spacy_sentencizer, transformer_tokenizer,
                                    max_chunk_qty=chunk_config['max_chunk_qty'], max_chunk_size=chunk_config['max_chunk_size'],
                                    heuristic_max_avg_tok_size=chunk_config['heuristic_max_avg_tok_size'])
                                      
    doc_chunks_merged = combine_adjacent_chunks(doc_chunks, 
                                                span=chunk_config['chunk_merge_qty'], stride=chunk_config['chunk_merge_stride'],
                                                chunk_sep=CHUNK_SEP)
    
    judge_doc_text_chunk_prefix = config.get('judge_doc_text_chunk_prefix', False)

    if rec is None:
        rec = dict(query_id=qid, doc_id=did,
                   human_relev=qrels[qid][did], 
                   human_relev_is_pos=qrels[qid][did] >= human_relev_grade_treshold,
                   query_text=query_text,
                   doc_text=doc_text,
                   doc_text_tok_len=len(transformer_tokenizer.tokenize(doc_text)),
                   merged_chunk_qty=len(doc_chunks_merged),
                   judged_chunks = [])
        
    rec['merged_chunk_qty']=len(doc_chunks_merged)

    for start, chunk_text in doc_chunks_merged:
        found = False
        for judged_chunk in rec['judged_chunks']:
            if judged_chunk['start'] == start:
                assert judged_chunk['chunk_text'] == chunk_text
                found = True
                break
        if found:
            continue

        response = judge_query_doc_pair(judge_gpt, qid, did, query_text, chunk_text)

        assert type(response) == list
        assert len(response) == 1
        assert type(response[0]) == dict
        rec['judged_chunks'].append(dict(start=start, 
                                    judgment=response[0]['judgment'],
                                    chunk_text=chunk_text,
                                    judge_full_response=response[0]))                                   
    
    if judge_doc_text_chunk_prefix:
        if not 'judged_doc_text_chunk_prefix' in rec:
            doc_text_chunk_prefix = '\n'.join([e[1] for e in doc_chunks])
            response = judge_query_doc_pair(judge_gpt, qid, did, query_text, doc_text_chunk_prefix)
            rec['judged_doc_text_chunk_prefix'] = dict(judgment=response[0]['judgment'],
                                        chunk_text=doc_text_chunk_prefix,
                                        judge_full_response=response[0])

    cache_mngr.put(qdoc_key, rec)

    return rec

def main(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    json_config_file_name = args.config
    with open(json_config_file_name, 'r') as json_file:
        config = json.load(json_file)

    collect_root=config['collect_root']
    dataset=config['dataset']
    input_subdir=config['input_subdir']
    output_subdir=config['output_subdir']

    gpt_model_name=config['gpt_model_name']

    logger.info(f'Collecting data for {dataset} dataset using {gpt_model_name} model')

    input_dir=f'{collect_root}/{dataset}/input_data/{input_subdir}/'
    output_dir=f'{collect_root}/{dataset}/{output_subdir}/{gpt_model_name}'
    cache_dir=f'{output_dir}/cache/'

    cache_mngr = CacheManagerJSON(cache_dir)

    queries = read_queries_dict(f'{input_dir}/{QUESTION_FILE_JSON}')
    qrels = read_qrels_dict(f'{input_dir}/{QREL_FILE}')

    # add Java JAR to the class path
    configure_classpath()

    from flexneuart.retrieval import create_featextr_resource_manager

    # create a resource manager
    resource_manager=create_featextr_resource_manager(resource_root_dir=f'{collect_root}/{dataset}/',
                                                    fwd_index_dir='forward_index')

    doc_index = resource_manager.getFwdIndex(config['doc_field_name'])

    load_dotenv()

    judge_gpt = GPTJudge(qrel="test_qrels", prompt_type="bing", engine=gpt_model_name)

    tokenizaton_config=config['tokenization']
    transformer_tokenizer = AutoTokenizer.from_pretrained(tokenizaton_config['tokenizer_model'])
    spacy_sentencizer=Sentencizer(tokenizaton_config['spacy_model'])

    qrel_id_list = sorted(list(qrels.keys()))

    np.random.seed(config['random_seed'])
    np.random.shuffle(qrel_id_list)

    human_relev_grade_treshold=config['human_relev_grade_treshold']

    query_dict_pos = defaultdict(list)
    query_dict_neg = defaultdict(list)

    for qid in qrel_id_list:
        for did, rel in qrels[qid].items():
            if rel >= human_relev_grade_treshold:
                query_dict_pos[qid].append(did)
            else:
                query_dict_neg[qid].append(did)

    query_doc_pos_pairs = interleave_query_docs(query_dict_pos)
    query_doc_neg_pairs = interleave_query_docs(query_dict_neg)

    query_doc_pos_pairs=query_doc_pos_pairs[0:config['pos_query_doc_pair_qty']]
    query_doc_neg_pairs=query_doc_neg_pairs[0:config['neg_query_doc_pair_qty']]

    # parnoid sanity check
    for qid, did in query_doc_pos_pairs:
        assert qrels[qid][did] >= human_relev_grade_treshold, f'{qid} {did}'
    for qid, did in query_doc_neg_pairs:
        assert qrels[qid][did] < human_relev_grade_treshold, f'{qid} {did}'


    output = []
    for qid, did in tqdm(query_doc_pos_pairs + query_doc_neg_pairs):
        output.append(proc_query_doc_pair(config, cache_mngr, doc_index, 
                                            queries, qrels, 
                                            spacy_sentencizer, transformer_tokenizer, judge_gpt,
                                            qid, did))
        
    with open(f'{output_dir}/output.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)

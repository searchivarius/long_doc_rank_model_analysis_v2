#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

USE_TRAIN=True


COLLECT_ROOT="/disk3/collections"
os.environ['COLLECT_ROOT']=COLLECT_ROOT
os.environ['TOKENIZERS_PARALLELISM']='false'


# In[ ]:


from flexneuart import configure_classpath
from flexneuart.config import QUESTION_FILE_JSON, QREL_FILE, DOCID_FIELD, TEXT_FIELD_NAME


# In[ ]:


configure_classpath()


# In[ ]:


from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.cand_provider import *
from flexneuart.retrieval.fwd_index import get_forward_index
from flexneuart.io.queries import read_queries
from flexneuart.io.qrels import read_qrels_dict


# In[ ]:


from flexneuart.io.utils import jsonl_gen

def read_queries_dict(file_name):
    """Read queries from a JSONL file and checks the document ID is set.

    :param file_name: an input file name
    :return: an dictionary where keys are query IDs and values are parsed query JSONs.
    """
    return {e[DOCID_FIELD] : e for e in jsonl_gen(file_name) }


# In[ ]:


MSMARCO_DOC_DIR=f'{COLLECT_ROOT}/msmarco_doc_v1_lb'
MSMARCO_PASS_DIR=f'{COLLECT_ROOT}/msmarco_pass_v1_lb'

if USE_TRAIN:
    # Let's do 5000
    QUERY_PART='train'
    #SAMPLE_SIZE=1000
    SAMPLE_SIZE=5000
else:
    QUERY_PART='dev_official'
    SAMPLE_SIZE=None


pass_queries = read_queries_dict(f'{MSMARCO_PASS_DIR}/input_data/{QUERY_PART}/{QUESTION_FILE_JSON}')
pass_qrels = read_qrels_dict(f'{MSMARCO_PASS_DIR}/input_data/{QUERY_PART}/{QREL_FILE}')

doc_queries = read_queries_dict(f'{MSMARCO_DOC_DIR}/input_data/{QUERY_PART}/{QUESTION_FILE_JSON}')
doc_qrels = read_qrels_dict(f'{MSMARCO_DOC_DIR}/input_data/{QUERY_PART}/{QREL_FILE}')


# In[ ]:


len(pass_qrels), len(doc_qrels)


# In[ ]:


doc_mngr=create_featextr_resource_manager(resource_root_dir=MSMARCO_DOC_DIR,
                                          fwd_index_dir='forward_index')
doc_prov=create_cand_provider(doc_mngr, PROVIDER_TYPE_LUCENE, f'lucene_index_text')
doc_text_indx=get_forward_index(doc_mngr, 'text_raw')


# In[ ]:


pass_mngr=create_featextr_resource_manager(resource_root_dir=MSMARCO_PASS_DIR,
                                          fwd_index_dir='forward_index')
#pass_prov=create_cand_provider(pass_mngr, PROVIDER_TYPE_LUCENE, f'lucene_index')
pass_text_indx=get_forward_index(pass_mngr, 'text_raw')


# In[ ]:


pass_queries_indexed_by_text = {e[TEXT_FIELD_NAME] : e for e in pass_queries.values()}
doc_queries_indexed_by_text = {e[TEXT_FIELD_NAME] : e for e in doc_queries.values()}
common_queries = set(pass_queries.keys()).intersection(set(doc_queries.keys()))


# In[ ]:


remove_qids = []
for qid in common_queries:
    if pass_queries[qid][TEXT_FIELD_NAME].strip() != doc_queries[qid][TEXT_FIELD_NAME].strip():
        remove_qids.append(qid)
common_queries = [qid for qid in common_queries if qid not in remove_qids]  
len(common_queries)


# In[ ]:


if SAMPLE_SIZE is not None:
    import numpy as np
    np.random.seed(0)
    sel_qids = np.random.choice(np.arange(len(common_queries)), SAMPLE_SIZE)
    common_queries = [common_queries[idx] for idx in sel_qids]


# In[ ]:


def replaceNonAlphaNumeric(s):
    return re.sub("[^0-9a-zA-Z]", " ", s)


# In[ ]:


from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


# Python3 implementation to print
# the longest common substring

# function to find and print
# the longest common substring of
# X[0..m-1] and Y[0..n-1]
def LCSSubStr(X: str, Y: str):
    m = len(X)
    n = len(Y)

    LCSuff = [[0 for i in range(n + 1)]
                for j in range(m + 1)]

    length = 0

    row, col = 0, 0
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    row = i
                    col = j
            else:
                LCSuff[i][j] = 0

    # if true, then no common substring exists
    if length == 0:
        return ""

    resultStr = ['0'] * length

    while LCSuff[row][col] != 0:
        length -= 1
        resultStr[length] = X[row - 1] # or Y[col-1]

        row -= 1
        col -= 1

    # required longest common substring
    return(''.join(resultStr))


# In[ ]:


# Dynamic Programming implementation of LCS problem

def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in range(m+1)]
    first = [[None]*(n+1) for i in range(m+1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
                first[i][j] = first[i-1][j-1] if L[i][j] > 1 else (i, j)
            else:
                if L[i-1][j] > L[i][j-1]:
                    L[i][j] = L[i-1][j]
                    first[i][j] = first[i-1][j]
                else:
                    L[i][j] = L[i][j-1]
                    first[i][j] = first[i][j-1]
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n], first[m][n]
#end of function lcs


# In[ ]:


def longest_common_substring(pass_text, doc_text,threshold = 0.8): 
    substring = LCSSubStr(pass_text,doc_text)
    similarity = len(substring) / len(pass_text)
    if similarity >= threshold:
        doc_ind = doc_text.index(substring)
        return similarity, doc_ind
    return None


# In[ ]:


import math
def longest_common_subsequence(pass_text,doc_text,output_threshold = 0.7, skip_threshold = 0.1, length_threshold = 1.2):
    max_lcs = 0
    max_indexes = (-1,-1)
    i = 0
    while i < len(doc_text):
        length, indexes = lcs(pass_text,doc_text[i : i + math.ceil(len(pass_text)*length_threshold)])
        if length > max_lcs:
            max_lcs = length
            doc_ind = indexes[1] + i
            max_indexes = (indexes[0],doc_ind)

        i += math.floor(len(pass_text)*skip_threshold)
    similarity = max_lcs / len(pass_text)
    if similarity >= output_threshold:
        return similarity, max_indexes[1]
    return None


# In[ ]:


import re
def from_docind_to_bertind(doc_text,doc_ind):
    bert_ind = 0
    for m in re.finditer(r'\S+', doc_text):
        w = m.group(0)
        if m.start() < doc_ind-1:
            bert_ind += len(tok.encode(w, add_special_tokens=False))
        else:
            if m.start() > doc_ind-1 and bert_ind > 0:
                bert_ind -= 1
            break
    return bert_ind


# In[ ]:


def matching_pipeline(passid, docid):
    #print('@@', passid, '->', docid)
    pass_text = pass_text_indx.get_doc_raw(passid)
    pass_text = replaceNonAlphaNumeric(pass_text)
    pass_text = pass_text.lower()
    
    doc_text = doc_text_indx.get_doc_raw(docid)
    doc_text = replaceNonAlphaNumeric(doc_text)
    doc_text = doc_text.lower()

    #print('##', len(pass_text), '->', len(doc_text))
    
    #pass length (# BERT tokens)
    pass_len_berttokens = len(tok.tokenize(pass_text))
    #doc length (# BERT tokens)
    doc_len_berttokens = len(tok.tokenize(doc_text))
    
    result = longest_common_substring(pass_text,doc_text)
    if result != None:
        similarity, doc_ind = result
        bert_ind = from_docind_to_bertind(doc_text,doc_ind)
        return {"matching method": "longest common substring", "similarity":similarity, "passage length (bert)": pass_len_berttokens,
               "document length (bert)": doc_len_berttokens, "match start (bert)" : bert_ind}
    
    
    result = longest_common_subsequence(pass_text, doc_text)
    if result != None:
        similarity, doc_ind = result
        bert_ind = from_docind_to_bertind(doc_text,doc_ind)
        return {"matching method": "longest common subsequence", "similarity":similarity, "passage length (bert)": pass_len_berttokens,
               "document length (bert)": doc_len_berttokens, "match start (bert)" : bert_ind}
    return None
    


# In[ ]:


from tqdm.auto import tqdm

qrel_matches = []
passids = set()

for qid in tqdm(common_queries):
    pass_ids = list(pass_qrels[qid].keys())
    doc_ids = list(doc_qrels[qid].keys())
    for passid in pass_ids:
        for docid in doc_ids:
            result = matching_pipeline(passid,docid)
            if result != None:
                result["qid"] = qid
                result["matching type"] = "QREL"
                result["passid"] = passid
                result["docid"] = docid
                qrel_matches.append(result)
    passids.update(pass_ids)
len(qrel_matches)


# In[ ]:


len(qrel_matches), len(passids)


# In[ ]:
passid_docid_pairs_qrel = [str(obj["passid"]) + "#" + str(obj["docid"]) for obj in qrel_matches]


# In[ ]:


#second matching type: use passage as query
import math
passids = list(passids)
retrieval_matches = []

#passid, docid that can also be matched in topk retrieval
passid_docid_pairs_qrel_retrieval = []

#top_k = 10
top_k = 5

for i in tqdm(range(len(passids))):  
    passid = passids[i]
    pass_text = pass_text_indx.get_doc_raw(passid)
    pass_text = replaceNonAlphaNumeric(pass_text)
    pass_text = pass_text.lower()
    
    res = run_text_query(doc_prov, top_k, pass_text)
    docids = [res[1][i].doc_id for i in range(len(res[1]))]

    #print(len(docids))
    
    for j in range(len(docids)):
        docid = docids[j]
        #print(docid)
        
        #if already matched in qrel, skip for efficiency
        if str(passid) + "#" + str(docid) in passid_docid_pairs_qrel:
            passid_docid_pairs_qrel_retrieval.append(str(passid) + "#" + str(docid))
            continue
            
        result = matching_pipeline(passid, docid)
        if result != None:
            result["qid"] = ""
            result["matching type"] = "RETRIEVAL"
            result["passid"] = passid
            result["docid"] = docid
            result["rank"] = j + 1
            retrieval_matches.append(result)
len(retrieval_matches)


# In[ ]:


retrieval_matches


# In[ ]:


import json

#combine qrel results with retrieval
qrel_retrieval_matches = []
for match in qrel_matches:
    key = match["passid"] + "#" + match["docid"]
    if key in passid_docid_pairs_qrel_retrieval:
        match["matching type"] = "QREL_AND_RETRIEVAL"
    qrel_retrieval_matches.append(match)

qrel_retrieval_matches += retrieval_matches   

len(qrel_retrieval_matches)


# In[ ]:


# update the rank for documents that are in QREL_AND_RETRIEVAL
def get_rank_helper(passid,docid):
    pass_text = pass_text_indx.get_doc_raw(passid)
    pass_text = replaceNonAlphaNumeric(pass_text)
    pass_text = pass_text.lower()
    
    res = run_text_query(doc_prov, top_k, pass_text)
    docids = [res[1][i].doc_id for i in range(len(res[1]))]
    for i in range(len(docids)):
        if docids[i] == docid:
            return i + 1

qrel_retrieval_matches_updated = []
for match in tqdm(qrel_retrieval_matches):
    if match["matching type"] == "QREL_AND_RETRIEVAL":
        rank = get_rank_helper(match["passid"], match["docid"])
        match["rank"] = rank  
    qrel_retrieval_matches_updated.append(match)
        
len(qrel_retrieval_matches_updated)


# In[ ]:


with open(f'final_results_{QUERY_PART}.txt', 'w') as outfile:
    json.dump(qrel_retrieval_matches, outfile)


# In[ ]:





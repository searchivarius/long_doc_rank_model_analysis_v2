
def judge_query_doc_pair(judge_gpt, qid, did, query_text, doc_text):
    input_dict = {
            "query": {"text": query_text, "qid": qid},
            "candidates": [
                {
                    "doc": {
                        "segment": doc_text,
                    },
                    "docid": did
                },
            ]
        }
        
    response = judge_gpt.judge(request_dict=input_dict)
    return response

def extract_judgment(response):
    assert type(response) == list
    assert len(response) == 1
    assert type(response[0]) == dict

    return response[0]['judgment']

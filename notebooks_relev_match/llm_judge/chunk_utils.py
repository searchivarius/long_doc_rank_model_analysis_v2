# This function can be fairly slow even with the exponential search enabled.
# However the main reason is not due tokenizing a lot of document prefixes,
# but due to splitting the document into sentences using Spacy.
# For this reason, it can be beneficial to heuristically truncate input documents
# based on the upper bound for the maximum number of characters for a specified 
# maximum number of tokens. This can be achieved by setting the parameter
# heuristic_max_avg_tok_size to a fairly large number, e.g, 10 or 20. In contrast,
# with, e.g., a BERT tokenizer, a typical token (which are often shorter than regular 
# English words) have 3-6 characters.
def extract_sentences_up_to_the_limit(text, spacy_sentencizer, tok, max_tok_qty,
                                      heuristic_max_avg_tok_size=None,
                                      use_exp_search=True):
    max_pref = ''

    if heuristic_max_avg_tok_size is not None:
        text = text[0: heuristic_max_avg_tok_size * max_tok_qty]
    
    sent_list=list(spacy_sentencizer(text))

    min_sent_id = 0
    max_sent_id = len(sent_list)
    
    if use_exp_search:
        max_sent_id = 1
    
        while max_sent_id < len(sent_list):
            if len(tok.tokenize(text[0:sent_list[max_sent_id].end_char])) > max_tok_qty:
                break  
            min_sent_id = max_sent_id
            max_sent_id = min(2*max_sent_id, len(sent_list))

    #print('Sent ID range:', min_sent_id, ' -> ', max_sent_id)
    
    for span in sent_list[min_sent_id:max_sent_id]:
        prefix = text[0:span.end_char]
        if len(tok.tokenize(prefix)) > max_tok_qty:
            break
        max_pref = prefix
    return max_pref

#
# This function takes a prefix of the text and splits it into at must chunk_qty chunks
# each of which has at most max_chunk_size tokens.
#
# This function can be fairly slow. However the main reason is not due tokenizing a lot of document prefixes,
# but due to splitting the document into sentences using Spacy.
#
# For this reason, it can be beneficial to heuristically truncate input documents
# based on the upper bound for the maximum number of characters for a specified 
# maximum number of tokens. This can be achieved by setting the parameter
# heuristic_max_avg_tok_size to a fairly large number, e.g, 10 or 20. In contrast,
# with, e.g., a BERT tokenizer, a typical token (which are often shorter than regular 
# English words) has about 3-5 characters on average.
#
def extract_n_chunks(text, spacy_sentencizer, transf_tokenizer, 
                     max_chunk_qty, max_chunk_size,
                     heuristic_max_avg_tok_size=None):
    tmp_arr = []
    max_tok_qty = max_chunk_qty * max_chunk_size 

    #print(max_chunk_qty, max_chunk_size, max_tok_qty)
    
    if heuristic_max_avg_tok_size is not None:
        text = text[0: heuristic_max_avg_tok_size * max_tok_qty]
    
    sent_list=list(spacy_sentencizer(text))
    max_pref = ""
    max_pref_max_tok_qty = max_chunk_size
    
    for span_idx, span in enumerate(sent_list):
        prefix = text[0:span.end_char]
        prefix_tok_qty = len(transf_tokenizer.tokenize(prefix))

        if prefix_tok_qty > max_pref_max_tok_qty or span_idx + 1 >= len(sent_list):
            #print('###', prefix_tok_qty, '@@', max_pref_max_tok_qty)            
            if max_pref:
                tmp_arr.append(max_pref)
                max_pref_max_tok_qty += max_chunk_size
                
            if prefix_tok_qty > max_tok_qty:
                break
        max_pref = prefix

    assert len(tmp_arr) <= max_chunk_qty
    if not tmp_arr:
        return []
    else:
        res = [(0, tmp_arr[0])]
        for k in range(1, len(tmp_arr)):
            prev_chunk = tmp_arr[k-1]
            prev_chunk_tok_qty = len(transf_tokenizer.tokenize(prev_chunk))
            res.append( (prev_chunk_tok_qty, tmp_arr[k][len(prev_chunk):]) )
        # Sanity check: all chunks should make a valid text prefix.
        assert text.startswith(''.join([e[1] for e in res]))
        return res

# chunks contains tuples (start, chunk text)
# combine adjacent chunks, discard incomplete training chunks
# unless it's the only chunk
def combine_adjacent_chunks(chunks, span, stride, chunk_sep = '\n'):
    res = []

    for k in range(0, len(chunks), stride):
        # Do not use incomplete chunks, unless it is the only chunk
        if res and k + span > len(chunks): break
        tmp_chunk_arr = [e[1] for e in chunks[k: k + span]]
        res.append( (chunks[k][0], chunk_sep.join(tmp_chunk_arr)) )

    return res


def interleave_query_docs(query_dict):
    """
    Create an interleaved array of (query_id, document_id) tuples.

    param: query_dict: Dictionary mapping query IDs to lists of document IDs.

    return: list: Interleaved list of (query_id, document_id) tuples.
    """
    # Find the maximum length of document lists
    # The trick of appending [] solves an issue with empty query_dict,
    # which will cause max to fail (it requires a non-empty sequence)
    max_len = max([len(docs) for docs in query_dict.values()] + [0])

    # Interleave tuples
    interleaved = []
    for position in range(max_len):
        for query_id, doc_ids in query_dict.items():
            if position < len(doc_ids):
                interleaved.append((query_id, doc_ids[position]))

    return interleaved

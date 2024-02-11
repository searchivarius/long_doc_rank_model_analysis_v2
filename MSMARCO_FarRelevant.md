# MS MARCO FarRelevant format description
Directory layout (explanation follows)
```
├── exper_desc
│   ├── bm25_lucene_test.json
│   └── lucene_conf.json
└── input_data
    ├── bitext
    │   ├── AnswerFields.jsonl.gz
    │   ├── QuestionFields.jsonl
    │   └── qrels.txt
    ├── dev_official
    │   ├── AnswerFields.jsonl.gz
    │   ├── QuestionFields.jsonl
    │   └── qrels.txt
    └── dev_official_sample1K
        ├── QuestionFields.jsonl
        └── qrels.txt -> ../dev_official/qrels.txt
```
The processed collection is divided into two parts:
1. training data: input_data/bitext
2. full testing data: input_data/dev_official
The paper, however, uses a 1K sample of dev_official, which is large enough.

Input data is JSONL format:
1. Queries are in QuestionFields.jsonl
2. Documents are in AnswerFields.jsonl.gz
3. Each line in a document/query file is a JSON entry with several "fields".
4. DOCNO denotes the identifier (query ID in a query file and document ID in a document file).
5. text_raw denotes the field with the original text, i.e., without tokenization or stop-word removal. 

TREC relevance (qrel) files are in the TREC qrel format: qrels.txt




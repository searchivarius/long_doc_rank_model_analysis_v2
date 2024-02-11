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

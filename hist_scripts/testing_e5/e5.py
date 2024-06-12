#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
    A convenience wrapper for sentence-BERT/sentence-transformer bi-encoder models.
    https://github.com/UKPLab/sentence-transformers
"""
import torch

from flexneuart import models
from flexneuart.models.base import BaseModel
from typing import List, Dict

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_position_ids(input_ids: Tensor, max_original_positions: int = 512, encode_max_length: int = 4096) -> Tensor:
    position_ids = list(range(input_ids.size(1)))
    factor = max(encode_max_length // max_original_positions, 1)
    if input_ids.size(1) <= max_original_positions:
        position_ids = [(pid * factor) for pid in position_ids]

    position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    return position_ids

INNER_MODEL_ATTR = 'model'

@models.register('biencoder_e5')
class BiEncoderE5(BaseModel):
    """
        A biencoder E5 wrapper class.
    """
    def __init__(self, model_name):
        super().__init__()
        assert model_name.startswith('dwzhu/e5')
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        setattr(self, INNER_MODEL_ATTR, model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def bert_param_names(self):
        # This returns more parameters than necessary, which means that the linear projection
        # layer will be updated using a small learning rate only. But, unfortunately, it is
        # hard to extract non-BERT specific parameters in a generic way. Plus, we would
        # really want to only fine-tune this model rather than train it from scratch.
        # Anyways, training from scratch would require support for in-batch negatives,
        # which we do not provide.
        return set([k for k in self.state_dict().keys() if k.startswith( f'{INNER_MODEL_ATTR}.')])

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:

        """
           "Featurizes" input. Convert input queries and texts to a set of features,
            which are compatible to the model's forward function.

            **ATTENTION!!!** This function *MUST* itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        query_qty = len(query_texts)
        assert query_qty == len(doc_texts)

        batch_dict_query = self.tokenizer(query_texts, max_length=min(max_query_len, 4096),
                                          padding=True, truncation=True, return_tensors='pt')
        batch_dict_query['position_ids'] = get_position_ids(batch_dict_query['input_ids'], max_original_positions=512,
                                                            encode_max_length=4096)

        batch_dict_doc = self.tokenizer(doc_texts, max_length=min(max_doc_len, 4096),
                                        padding=True, truncation=True, return_tensors='pt')
        batch_dict_doc['position_ids'] = get_position_ids(batch_dict_doc['input_ids'], max_original_positions=512,
                                                            encode_max_length=4096)


        return batch_dict_query['input_ids'], batch_dict_query['attention_mask'], batch_dict_query['position_ids'], \
               batch_dict_doc['input_ids'], batch_dict_doc['attention_mask'], batch_dict_doc['position_ids']

    def encode(self, input_ids, attention_mask, position_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        embeddings = average_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, query_input_ids, query_attention_mask, query_position_ids,
                       doc_input_ids, doc_attention_mask, doc_position_ids):
        query_embed = self.encode(input_ids=query_input_ids, attention_mask=query_attention_mask, position_ids=query_position_ids)
        doc_embed = self.encode(input_ids=doc_input_ids, attention_mask=doc_attention_mask, position_ids=doc_position_ids)

        return torch.sum(doc_embed * query_embed, dim=-1)

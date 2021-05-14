import os
import re
import math
import multiprocessing as mp

import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

import transformers
from transformers import BertModel, BertTokenizer

import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def split_dataframe(max_len, sub, doc):
    split_doc = doc.split()
    res_text = [" ".join(split_doc[i:i + max_len]) for i in range(0, len(doc), max_len)]
    #res_text.append("[DOCEND]")
    res_targ = [sub for i in range(0, len(doc), max_len)]
    #res_targ.append(-1)
    return (res_text, res_targ)

# A simple data structure for representing multilabel datasets (taken from A8)
class MultiLabelDataset(torch.utils.data.Dataset):

    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.targets = labels
        self.max_len = max_len
        self.tokens = []

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.tokens(index)
        
    def __pad_all(self):
        pass

    def __tokenize(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
            'targets': torch.tensor(self.targets[index])
        }

class MultiLabelChunkedDataset(MultiLabelDataset):
    '''Represents a multi-label dataset for single documents longer the maximum length of the model.'''
    def __init__(self, df, tokenizer, max_len):
        self.split_dataframe = split_dataframe
        # Split samples into sizes that will fit in our model, leaving plenty of room for special tokens
        chunks, labels = self.__load_dataframe(df, max_len - 10)
        
        super(MultiLabelChunkedDataset, self).__init__(chunks, labels, tokenizer, max_len)

    def __getitem__(self, index):
        return (self.text[index], self.targets[index])

    def index_of(self, text):
        return self.text.index(text)

    def __load_dataframe(self, df, max_len):
        df.subject = df.subject.astype("category").cat.codes
        self.result_text = []
        self.result_targets = []
        pool = mp.Pool(mp.cpu_count() - 1)

        print("DataFrame loader: Generating size {} chunks across {} processes...".format(math.ceil(df.shape[0] / (mp.cpu_count() - 1)), mp.cpu_count() - 1))

        prepared_data = df.values.tolist()
        prepared_data = [[max_len, sub, doc] for sub, doc in prepared_data]

        result = pool.starmap(split_dataframe, prepared_data, chunksize=int(df.shape[0] / mp.cpu_count() - 1))

        pool.close()
        pool.join()

        for r in result:
            self.result_text.append(r[0])
            self.result_targets.append(r[1])

        return self.result_text, self.result_targets

    @staticmethod
    def tokenize(text, targets, tokenizer, max_len):
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True
        )
        ids = [inputs['input_ids']]
        mask = [inputs['attention_mask']]
        token_type_ids = [inputs["token_type_ids"]]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
            'targets': torch.tensor([targets])
        }

# Represents an instance of BERT (taken from A8)
class BERTClass(torch.nn.Module):
    def __init__(self, NUM_OUT):
        super(BERTClass, self).__init__()
                   
        self.l1 = BertModel.from_pretrained("bert-base-uncased")
#         self.pre_classifier = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Linear(768, NUM_OUT)
#         self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = torch.nn.Tanh()(pooler)
#         pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output

if __name__ == "__main__":
    # Test multilabel chunked dataset loading
    data = pd.DataFrame(pd.read_pickle(open("clean_data/transformer_ready_data_1229.p", "rb")))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    training_data = MultiLabelChunkedDataset(data, tokenizer, 512)

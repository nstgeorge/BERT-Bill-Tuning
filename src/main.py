###
#   Transformer Fine-Tuning for Congressional Bill Classification by Industry
#   Nate St. George
#   Created for CS 436 (Natural Language Processing)
###

import time
import os

from bert_training import BERTClass, MultiLabelChunkedDataset

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import transformers
from transformers import BertTokenizer
import progressbar

import torch
from torch import cuda

MAX_LEN = 128
BATCH_SIZE = None
EPOCHS = 30
LEARNING_RATE = 2e-02

DATA_PATH = "clean_data/transformer_ready_data_1229.p"
TRAIN_SIZE = 1000

if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'

    print("Using {} for fine-tuning task.".format(device))

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    data = pd.DataFrame(pd.read_pickle(open(DATA_PATH, "rb")))

    # train_labels = data.subject[:TRAIN_SIZE].astype("category").cat.codes
    # train_texts = data.content[:TRAIN_SIZE]
    # test_labels = data.subject[TRAIN_SIZE:].astype("category").cat.codes
    # test_texts = data.content[TRAIN_SIZE:]

    train_data = pd.DataFrame(data[:TRAIN_SIZE])
    test_data = pd.DataFrame(data[TRAIN_SIZE:])

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)

    def train(model, training_loader, optimizer, accumulation_steps):
        '''Train the model.'''
        model.train()

        for doc_data in progressbar.progressbar(training_loader):
            doc_loss = torch.zeros(1).requires_grad_().cuda()
            for entry in doc_data:
                ids = entry['ids'].to(device, dtype = torch.long)
                mask = entry['mask'].to(device, dtype = torch.long)
                token_type_ids = entry['token_type_ids'].to(device, dtype = torch.long)
                targets = entry['targets'].to(device, dtype = torch.long)

                outputs = model(ids, mask, token_type_ids)

                doc_loss = loss_fn(outputs, targets).item() + doc_loss

            loss = doc_loss / len(doc_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss
        
    def validation(model, testing_loader):
        '''Validate the model.'''
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for doc_data in progressbar.progressbar(testing_loader):
                doc_outputs = torch.zeros(NUM_OUT)
                
                for entry in doc_data:
                    targets = entry['targets']

                    ids = entry['ids'].to(device, dtype = torch.long)
                    mask = entry['mask'].to(device, dtype = torch.long)
                    token_type_ids = entry['token_type_ids'].to(device, dtype = torch.long)

                    outputs = model(ids, mask, token_type_ids)
                    outputs = torch.sigmoid(outputs).cpu().detach()

                    doc_outputs = (outputs / len(doc_data)) + doc_outputs

                fin_outputs.extend(doc_outputs)
                fin_targets.extend(targets)
        return torch.stack(fin_outputs), torch.stack(fin_targets)

    def collate_by_document(batch):
        '''Create batches for BERT based on document length.'''
        # If batch size is greater than 1, split up tokenization
        if isinstance(batch[0], tuple):
            result = []
            for sub_batch in batch:
                result.extend(tokenize_across_batch(sub_batch))
            return result
        return tokenize_across_batch(batch)
        

    def tokenize_across_batch(batch):
        result = []
        for index, text in enumerate(batch[0]):
            result.append(MultiLabelChunkedDataset.tokenize(text, batch[1][index], tokenizer, MAX_LEN))
        return result

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    NUM_OUT = len(data.subject.unique())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Splitting data into chunks...")

    training_data = MultiLabelChunkedDataset(train_data, tokenizer, MAX_LEN)
    test_data = MultiLabelChunkedDataset(test_data, tokenizer, MAX_LEN)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    #'drop_last': True,
                    'collate_fn': collate_by_document,
                    'num_workers': 0
                    }

    test_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    #'drop_last': True,
                    'collate_fn': collate_by_document,
                    'num_workers': 0
                    }    

    training_loader = torch.utils.data.DataLoader(training_data, **train_params)
    testing_loader = torch.utils.data.DataLoader(test_data, **test_params)

    # Create and train model

    model = BERTClass(NUM_OUT)
    model.to(device)    

    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

    print("Training with {} samples.".format(len(training_data)))

    for epoch in range(EPOCHS):
        loss = train(model, training_loader, optimizer, 10)
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')  
        guess, targs = validation(model, testing_loader)
        guesses = torch.max(guess, dim=1)
        targets = torch.max(targs, dim=0)
        print(guesses, targets)
        print('Accuracy on test set: {}'.format(accuracy_score(guesses.indices, targs)))
    
    if not os.path.exists('models'):
        os.makedirs('models')

    torch.save(model.state_dict(), open("models/model_{}.p".format(int(time.time())), "wb"))
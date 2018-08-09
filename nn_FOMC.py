import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import operator
import os, math
import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
#os.chdir("C:\\Users\\jooho\\NLPProject")

os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

POS_LABEL = 1
NEG_LABEL = -1
NONE_LABEL = 0 

def load_data():
    df = pd.read_pickle("all_data.pickle")
    return df

def make_train_test_data(df, label, test_size = 0.2):
    X = df.statements
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def make_vocab(statements):
        vocab = Counter([word for content in statements for word in content])
        word_to_idx = {k: v+1 for v, k in enumerate(vocab)} # word to index mapping
        word_to_idx["UNK"] = 0 # all the unknown words will be mapped to index 0
        vocab = set(word_to_idx.keys())
        return vocab, word_to_idx

class TextClassificationDataset(tud.Dataset):
    '''
    PyTorch provide a common dataset interface. 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    The dataset encodes documents into indices. 
    With the PyTorch dataloader, you can easily get batched data for training and evaluation. 
    '''
    def __init__(self, statements, labels, vocab, word_to_idx):
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}       
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1, NONE_LABEL : 2}
        self.idx_to_label = [POS_LABEL, NEG_LABEL, NONE_LABEL]
        self.vocab_size = len(self.vocab)
        self.statements = statements
        self.labels = labels
        
    def __len__(self):
        return len(self.statements)
    
    
    def __getitem__(self, idx):
        item = np.zeros(self.vocab_size)
        item = torch.from_numpy(item)
        if len(self.labels) > 0: # in training or evaluation, we have both the document and label
            for word in self.statements[idx]:
                item[self.word_to_idx.get(word, 0)] += 1
            label = self.label_to_idx[self.labels[idx]]
            return item, label
        else: # in testing, we only have the document without label
            for word in self.statements[idx]:
                item[self.word_to_idx.get(word, 0)] += 1
            return item    

    def make_vectors(self):
        X = []
        y = []
        for i in range(len(self.statements)):        
            X.append(np.array(self[i][0]))
            y.append(self[i][1])
        return np.array(X),np.array(y)
        
df = load_data()
label = 'tnx_buckets_1d'
vocab, word_to_idx = make_vocab(df.statements)
train_data, test_data, train_labels, test_labels = make_train_test_data(df, label)              
train_dataset = TextClassificationDataset(train_data, train_labels, vocab, word_to_idx)
X,y = train_dataset.make_vectors()
test_dataset = TextClassificationDataset(test_data, test_labels, vocab, word_to_idx)
X_test,y_test = test_dataset.make_vectors()
nnet = MLPClassifier(activation = 'relu', max_iter = 1000,hidden_layer_sizes=(15,2), solver = 'adam')
nnet.fit(X,y)
preds = nnet.predict(X_test)
(preds == y_test).sum()/len(preds)

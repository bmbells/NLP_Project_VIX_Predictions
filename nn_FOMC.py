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
from gensim.models import Word2Vec
#os.chdir("C:\\Users\\jooho\\NLPProject")

os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

POS_LABEL = 1
NEG_LABEL = -1
NONE_LABEL = 0 

def load_data():
    """Load all necessary data"""
    df = pd.read_pickle("all_data.pickle")
    return df

def word_to_vec(df):
    all_sentences = []
    for i in range(len(df)):
        for j in range(len(df.sentences[i])):
            all_sentences.append(df.sentences[i][j])
    model = Word2Vec(all_sentences, min_count = 0, window=5) #unsure best window 
    model.train(all_sentences, total_examples=len(all_sentences), epochs = 10)
    return model

def turn_statements_to_vectors(df, model):
    all_statements = []
    for i in range(len(df.statements)):
        all_vectors = []
        for j in range(len(df.statements[i])):
            vec = model.wv[df.statements[i][j]]
            all_vectors.append(vec)
        all_statements.append(np.sum(all_vectors, axis = 0))   
    return np.array(all_statements)    
            

def make_train_test_data(df, model, label, test_size = 0.2):
    """Divide data into train and test data for model training and validating."""
    X = turn_statements_to_vectors(df, model)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)#, random_state=42)
    return X_train, X_test, y_train, y_test

def make_vocab(statements):
        vocab = Counter([word for content in statements for word in content])
        word_to_idx = {k: v+1 for v, k in enumerate(vocab)} # word to index mapping
        word_to_idx["UNK"] = 0 # all the unknown words will be mapped to index 0
        vocab = set(word_to_idx.keys())
        return len(vocab), word_to_idx

class TextClassificationDataset(tud.Dataset):
    '''
    PyTorch provide a common dataset interface. 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    The dataset encodes documents into indices. 
    With the PyTorch dataloader, you can easily get batched data for training and evaluation. 
    '''
    def __init__(self, statements, labels, vocab_size, word_to_idx):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}       
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1, NONE_LABEL : 2}
        self.idx_to_label = [POS_LABEL, NEG_LABEL, NONE_LABEL]
        self.vocab_size = vocab_size
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

def main():
    df = load_data()
    vocab_size, word_to_idx = make_vocab(df.statements)
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    model = word_to_vec(df)
    scores = []
    for label in labels:
        train_data, test_data, train_labels, test_labels = make_train_test_data(df, model, label)
        #train_dataset = TextClassificationDataset(train_data, train_labels, vocab_size, word_to_idx)
        #X,y = train_dataset.make_vectors()
        #test_dataset = TextClassificationDataset(test_data, test_labels, vocab_size, word_to_idx)
        #X_test,y_test = test_dataset.make_vectors()
        X = train_data
        y = train_labels
        nnet = MLPClassifier(activation = 'logistic',hidden_layer_sizes=(50,2), solver = 'lbfgs', alpha = .1)
        nnet.fit(X,y)
        preds = nnet.predict(test_data)
        score = (preds == test_labels).sum()/len(preds)
        print(f"Score for {label} = {score}")
        scores.append(score)
    return scores
    
if __name__ == '__main__':        
    NUM_EPOCHS = 10
    all_scores = []
    for i in range(NUM_EPOCHS):
        print(i)
        all_scores.append(main())
    all_scores = np.array(all_scores)    
    print(np.mean(all_scores,axis= 0))

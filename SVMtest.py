import pandas as pd
import os
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve as vc
from sklearn.svm import SVC
from collections import Counter, defaultdict

os.chdir("C:\\Users\\jooho\\NLPProject\\NLP_Project_VIX_Predictions")

df = pd.read_pickle('preprocessed_minutes.pickle')
VOCAB_SIZE = 500

vocab = Counter([word for data in df['statements'] for word in data]).most_common(VOCAB_SIZE)
word_to_idx = {v[0]: k for k, v in enumerate(vocab)}
vocab_size = len(word_to_idx.keys())


def onehotencode(lst, word_to_idx, vocab_size):
    item = np.zeros(vocab_size)
    for word in lst:
        if word in word_to_idx.keys() and item[word_to_idx[word]]==0:
            item[word_to_idx[word]]+=1
    return item

df['BoW'] = df.statements.apply(onehotencode, word_to_idx = word_to_idx, vocab_size = vocab_size)

def make_train_test_data(df, label, test_size = .25):
    X = np.array(df.BoW.tolist())
    y = df[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = test_size, random_state = 42)
    return X_train, X_test, Y_train, Y_test

C_param = [.0001, .001, .01, .1,1,10,100] #Values for C
buckets = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
kernels = ['linear', 'rbf']
numtests = 3

for bucket in buckets:
    for kernel in kernels:
        train_scores, val_scores = vc(SVC(kernel = kernel), np.array(df.BoW.tolist()),df[bucket],param_name='C',param_range= C_param, cv=5, scoring='accuracy' )
        avg_val_scores = np.mean(val_scores, axis=1)
        Cindex = np.argmax(avg_val_scores)
        maxC = C_param[Cindex]
        print ('For SVC of {} with a {} kernel, C value of {} had best average Validation Score of {:.3f}.'.format(bucket, kernel, maxC, avg_val_scores[Cindex]))
        tests=[]
        for i in range(numtests):
            clf = SVC(C=maxC, kernel = kernel)
            X_train, X_test, Y_train, Y_test = make_train_test_data(df, bucket)
            clf.fit(X_train, Y_train)
            tests.append(clf.score(X_test, Y_test))
        print ('For SVC of {} with a {} kernel and C value of {}, model achieved test score of {:.3f}.\n'.format(bucket, kernel, maxC, np.mean(tests)))

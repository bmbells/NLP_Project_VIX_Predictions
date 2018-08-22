import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

POS_LABEL = 1
NEG_LABEL = -1
NONE_LABEL = 0 

def load_data():
    """Returns pre-processed dataframe of statements and labels"""
    
    df = pd.read_pickle("all_data.pickle")
    return df

def word_to_vec(df):
    """Create the word2vec model to convert words to vectors"""
    
    all_sentences = []
    for i in range(len(df)):
        for j in range(len(df.sentences[i])):
            all_sentences.append(df.sentences[i][j])
    model = Word2Vec(all_sentences, min_count = 0, window=5) #unsure best window 
    model.train(all_sentences, total_examples=len(all_sentences), epochs = 10)
    return model

def turn_statements_to_vectors(df, model):
    """
    Convert the statements to vectors by summing up all word embedding vectors
    for each statement. 
    
    Returns a list of all statements in vector form.
    """
    all_statements = []
    for i in range(len(df.sentences)):
        all_vectors = []
        for j in range(len(df.sentences[i])):
            for k in range(len(df.sentences[i][j])):
                vec = model.wv[df.sentences[i][j][k]]
                all_vectors.append(vec)
        all_statements.append(np.sum(all_vectors, axis = 0))   
    return np.array(all_statements)    
            

def make_train_test_data(df, model, label):
    """Divide data into train and test data for model training and validating."""
    
    train_data = df[df.set == "train"]
    test_data = df[df.set == "test"]
    X_train = turn_statements_to_vectors(train_data, model)
    X_test = turn_statements_to_vectors(test_data, model)
    y_train = train_data[label]
    y_test = test_data[label]
    return X_train, X_test, y_train, y_test

def param_tuning(train_data, train_labels, label):
    """
    Function used when parameter tuning before running the model. Performs
    cross-validation to pick the best performing activation, layer size, and
    number of layers for the training data and given test.
    
    """
    score = np.inf * -1
    activations = ['logistic', 'identity']
    hidden_layer_sizes = [10,50,100]
    nlayers = [1,2]
    X = train_data
    y = train_labels
    act_good = None
    nl_good = None
    hls_good = None
    kf = KFold(n_splits = 5)
    for act in activations:
        for nl in nlayers:
                for hls in hidden_layer_sizes:
                    kfold_scores = []
                    for train, test in kf.split(train_data):
                        nnet = MLPClassifier(activation = act ,hidden_layer_sizes=(hls,nl), solver = 'lbfgs', alpha = .1)
                        nnet.fit(X[train],y[train])
                        preds = nnet.predict(X[test])
                        score = (preds == y[test]).sum()/len(preds)
                        kfold_scores.append(score)
                    print(f"Score for {label} {act} {hls} {nl} = {np.mean(kfold_scores)}")
                    if np.mean(kfold_scores) > score:
                        act_good = act
                        nl_good = nl
                        hls_good = hls
                print()
    return act_good, nl_good, hls_good                   
                
def main():
    """
    Create NN for each test and output the accuracy score on the out of sample data.
    """
    
    df = load_data()
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    model = word_to_vec(df)
    scores = []
    for label in labels:
        train_data, test_data, train_labels, test_labels = make_train_test_data(df, model, label)
        X = train_data
        y = train_labels
        act, nl, hls = param_tuning(train_data, train_labels, label)
        nnet = MLPClassifier(activation = act,hidden_layer_sizes=(hls,nl), solver = 'lbfgs', alpha = .1)
        nnet.fit(X,y)
        preds = nnet.predict(test_data)
        score = (preds == test_labels).sum()/len(preds)
        print(f"Score for {label} = {score}")
        print()
        scores.append(score)
    return scores
    
if __name__ == '__main__':       
    scores = main()

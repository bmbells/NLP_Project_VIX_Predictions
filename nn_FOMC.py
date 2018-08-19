import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    """Create the word2vec model to convert words to vectors"""
    all_sentences = []
    for i in range(len(df)):
        for j in range(len(df.sentences[i])):
            all_sentences.append(df.sentences[i][j])
    model = Word2Vec(all_sentences, min_count = 0, window=5) #unsure best window 
    model.train(all_sentences, total_examples=len(all_sentences), epochs = 10)
    return model

def turn_statements_to_vectors(df, model):
    """Convert the statements to vectors"""
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

def param_tuning():
    """Function used when parameter tuning before running the model. I ran some 
    manual experiments using this code."""
    df = load_data()
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    model = word_to_vec(df)
    scores = []
    activations = ['relu', 'logistic', 'identity']
    hidden_layer_sizes = [10,50,100]
    nlayers = [1,2]
    alphas = [.0001, .001, .01, .1, .5]
    epochs = 50
    for label in labels:
            train_data, test_data, train_labels, test_labels = make_train_test_data(df, model, label)
            X = train_data
            y = train_labels
            #check out parameters
            for nl in nlayers:
                for hls in hidden_layer_sizes:
                    scores = []
                    for i in range(epochs):
                        alpha = .1
                        act = 'logistic'
                        nnet = MLPClassifier(activation = act ,hidden_layer_sizes=(hls,nl), solver = 'lbfgs', alpha = alpha)
                        nnet.fit(X,y)
                        preds = nnet.predict(test_data)
                        score = (preds == test_labels).sum()/len(preds)
                        scores.append(score)
                    print(f"Score for {label} {act} {hls} {nl} {alpha} = {np.mean(scores)}")
                
def main():
    """Create NN for each test and output the accuracy score on the out of sample data"""
    df = load_data()
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    model = word_to_vec(df)
    scores = []
    for label in labels:
        train_data, test_data, train_labels, test_labels = make_train_test_data(df, model, label)
        X = train_data
        y = train_labels
        nnet = MLPClassifier(activation = 'logistic',hidden_layer_sizes=(50,1), solver = 'lbfgs', alpha = .1)
        nnet.fit(X,y)
        preds = nnet.predict(test_data)
        score = (preds == test_labels).sum()/len(preds)
        print(f"Score for {label} = {score}")
        scores.append(score)
    return scores
    
if __name__ == '__main__':       
    """Run the main function multiple times and print the average scores"""
    NUM_EPOCHS = 50
    all_scores = []
    for i in range(NUM_EPOCHS):
        print(i)
        all_scores.append(main())
    all_scores = np.array(all_scores)    
    print(np.mean(all_scores,axis= 0))

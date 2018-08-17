import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve as vc

os.chdir("C:\\Users\\jooho\\NLPProject\\NLP_Project_VIX_Predictions")

def load_data():
    """Returns pre-processed dataframe of statements and labels"""

    df = pd.read_pickle("all_data.pickle")
    return df

def sumembeds(lst, model, EMBEDSIZE):
    """Takes word embeddings from Word2Vec model and creates input by summing through words"""

    item = np.zeros(EMBEDSIZE)
    for word in lst: #Sum through word embeddings for each word
        item+=model.wv[word]
    return item

def create_input_embeddings(embedSIZE):
    """Adds SumEmbeds column to dataframe

    SumEmbeds is sum of word embeddings for words in df.statments. Used as input for Random Forest.
    """

    df = load_data()
    all_sentences = []
    for i in range(len(df)):
        for j in range(len(df.sentences[i])):
            all_sentences.append(df.sentences[i][j])
    model = Word2Vec(all_sentences,size=embedSIZE, min_count = 0, window=5)
    model.train(all_sentences, total_examples=len(all_sentences), epochs = 10)
    df['SumEmbeds'] = df.statements.apply(sumembeds, model=model, EMBEDSIZE = embedSIZE) #Create Continuous BoW as input

    return df

def make_train_test_data(df, label, test_size = .25):
    """ Randomly seperates data into train and test sets"""

    X = np.array(df.SumEmbeds.tolist())
    y = df[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = test_size)
    return X_train, X_test, Y_train, Y_test

def run_RFC(df, n_estimators, buckets, max_depth, numtests):
    """ Runs RandomForestClassifier for each test bucket

    Uses cross validation to pick best max_depth for number of trees in forest. Train RandomForestClassifier and test multiple times, average of test score is reported test score.
    """

    for bucket in buckets:
        for estimator in n_estimators:
            X_train, X_test, Y_train, Y_test = make_train_test_data(df, bucket)
            train_scores, val_scores = vc(RFC(n_estimators = estimator), X_train,Y_train,param_name='max_depth',param_range= max_depth, cv=5, scoring='accuracy' )
            avg_val_scores = np.mean(val_scores, axis=1) #Take average of validation scores
            index = np.argmax(avg_val_scores) #Determine which max_depth had highest val score
            maxDEPTH = max_depth[index]
            print ('For Random Forest of {} with {} estimators, max depth of {} had best average Validation Score of {:.3f}.'.format(bucket, estimator, maxDEPTH, avg_val_scores[index]))
            tests=[]
            for i in range(numtests): #Will train and test multiple times
                clf = RFC(n_estimators=estimator, max_depth = maxDEPTH)
                clf.fit(X_train, Y_train)
                tests.append(clf.score(X_test, Y_test))
            print ('For Random Forest of {} with {} estimators and max depth of {}, model achieved test score of {:.3f}.\n'.format(bucket, estimator, maxDEPTH, np.mean(tests)))

def main():
    """ Test FOMC data with RandomForestClassifier"""

    df = create_input_embeddings(embedSIZE = 100)
    n_estimators = [10,25,50] #Number of trees in Random Forest
    buckets = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    max_depth = [5,10,15,20] #Max depth options
    numtests = 20 #Train and test 20 times
    run_RFC(df, n_estimators=n_estimators, buckets=buckets, max_depth=max_depth, numtests=numtests)


if __name__ == '__main__':
    main()

import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier as RFC
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

def make_train_test_data(df, label):
    """ Split data into train and test sets"""

    df_train = df[df.set == "train"]
    df_test = df[df.set == "test"]
    X_train = np.array(df_train.SumEmbeds.tolist())
    X_test = np.array(df_test.SumEmbeds.tolist())
    Y_train = df_train[label]
    Y_test = df_test[label]
    return X_train, X_test, Y_train, Y_test

def run_RFC(df, buckets, feat_examined, max_depth):
    """ Runs RandomForestClassifier for each test bucket

    Uses cross validation to pick best max_depth for number of trees in forest. Train RandomForestClassifier and test multiple times, average of test score is reported test score.
    """

    for bucket in buckets:
        featScore = []
        maxDepthScore = []
        for frac in feat_examined:
            X_train, X_test, Y_train, Y_test = make_train_test_data(df, bucket)
            train_scores, val_scores = vc(RFC(n_estimators = 25, max_features = frac), X_train,Y_train,param_name='max_depth',param_range= max_depth, cv=3, scoring='accuracy' )
            avg_val_scores = np.mean(val_scores, axis=1) #Take average of validation scores
            index = np.argmax(avg_val_scores) #Determine which max_depth had highest val score
            maxDepthScore.append(max_depth[index]) #Append best max depth
            featScore.append(avg_val_scores[index]) #Append validation score
        featidx = np.argmax(featScore) #Pick best validation score
        bestfeat = feat_examined[featidx]
        bestdepth = maxDepthScore[featidx]
        print ('For Random Forest of {}, max depth of {} and {} percent of features examined had best average Validation Score of {:.3f}.'.format(bucket, bestdepth, bestfeat*100, featScore[featidx]))
        tests=[]
        clf = RFC(n_estimators=25, max_depth = bestdepth, max_features = bestfeat)
        clf.fit(X_train, Y_train)
        print ('For Random Forest of {} with max depth of {} and {} percent of features, model achieved Test Score of {:.3f}.\n'.format(bucket, bestdepth, bestfeat*100, clf.score(X_test, Y_test)))

def main():
    """ Test FOMC data with RandomForestClassifier"""

    df = create_input_embeddings(embedSIZE = 100)
    buckets = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    feat_examined = [.05,.1,.2,.5,1]
    max_depth = [5,10,15,20] #Max depth options
    run_RFC(df, buckets=buckets, feat_examined = feat_examined,max_depth=max_depth)


if __name__ == '__main__':
    main()

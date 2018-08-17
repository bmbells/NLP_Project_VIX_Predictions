import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve as vc
from sklearn.svm import SVC
from collections import Counter, defaultdict

os.chdir("C:\\Users\\jooho\\NLPProject\\NLP_Project_VIX_Predictions")

def create_input(VOCAB_SIZE):
    """Adds BoW column to dataframe

    BoW is Bag of Words for words in df.statements. Used as input for SVM.
    """

    df = pd.read_pickle('all_data.pickle')
    vocab = Counter([word for data in df['statements'] for word in data]).most_common(VOCAB_SIZE) #Find 500 most common words
    word_to_idx = {v[0]: k for k, v in enumerate(vocab)}
    idx_to_word = {v:k for k, v in word_to_idx.items()}
    vocab_size = len(word_to_idx.keys())
    df['BoW'] = df.statements.apply(onehotencode, word_to_idx = word_to_idx, vocab_size = vocab_size) #Create Bag of Words

    return df, idx_to_word

def onehotencode(lst, word_to_idx, vocab_size):
    """ Creates Bag of Words for input"""

    item = np.zeros(vocab_size)
    for word in lst:
        if word in word_to_idx.keys() and item[word_to_idx[word]]==0: #Use Binary BoW, values of 1s and 0s
            item[word_to_idx[word]]+=1
    return item

def make_train_test_data(df, label, test_size = .25):
    """ Randomly seperates data into train and test sets"""

    X = np.array(df.BoW.tolist())
    y = df[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = test_size)
    return X_train, X_test, Y_train, Y_test

def run_SVM(df, C_param, buckets, kernels, numtests, idx_to_word):
    """ Runs SupportVectorMachine for each test bucket

    Uses cross validation to pick best C paramater for given kernel in SVM. Train SVM and test multiple times, average of test score is reported test score.
    """

    for bucket in buckets:
        for kernel in kernels:
            X_train, X_test, Y_train, Y_test = make_train_test_data(df, bucket)
            train_scores, val_scores = vc(SVC(kernel = kernel), X_train,Y_train,param_name='C',param_range= C_param, cv=5, scoring='accuracy' )
            avg_val_scores = np.mean(val_scores, axis=1) #Take average of validation scores
            Cindex = np.argmax(avg_val_scores) #Determine which C parameter had highest val score
            maxC = C_param[Cindex]
            print ('For SVC of {} with a {} kernel, C value of {} had best average Validation Score of {:.3f}.'.format(bucket, kernel, maxC, avg_val_scores[Cindex]))
            tests=[]
            weights = np.zeros((3,500))
            for i in range(numtests): #Will train and test multiple times
                clf = SVC(C=maxC, kernel = kernel)
                clf.fit(X_train, Y_train)
                if kernel == 'linear':
                    weights+= clf.coef_
                tests.append(clf.score(X_test, Y_test))
            print ('For SVC of {} with a {} kernel and C value of {}, model achieved test score of {:.3f}.\n'.format(bucket, kernel, maxC, np.mean(tests)))
            if kernel == 'linear':
                avgweights = weights/numtests
                high_down_weights = avgweights[0,:].argsort()[-10:][::-1]
                high_unch_weights = avgweights[1,:].argsort()[-10:][::-1]
                high_up_weights = avgweights[2,:].argsort()[-10:][::-1]
                down_words = [idx_to_word[idx] for idx in high_down_weights]
                unch_words = [idx_to_word[idx] for idx in high_unch_weights]
                up_words = [idx_to_word[idx] for idx in high_up_weights]
                print ('Highest Weight Down Words: ', down_words)
                print ('Highest Weight Unch Words: ', unch_words)
                print ('Highest Weight Up Words: ', up_words)

def main():
    """ Test FOMC data with SupportVectorMachine"""

    df, idx_to_word = create_input(VOCAB_SIZE=500)
    C_param = [.0001, .001, .01, .1,1,10,100] #Values for C
    buckets = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    kernels = ['linear', 'rbf'] #Kernel function
    numtests = 20 #Train and test 20 times
    run_SVM(df = df, C_param = C_param, buckets=buckets, kernels=kernels, numtests=numtests, idx_to_word = idx_to_word)

if __name__ == '__main__':
    main()

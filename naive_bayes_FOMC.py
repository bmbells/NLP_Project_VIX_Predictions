import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import operator
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
#os.chdir("C:\\Users\\jooho\\NLPProject")
os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

POS_LABEL = 1
NEG_LABEL = -1
NONE_LABEL = 0   

def load_data():
    """Load all necessary data"""
    df = pd.read_pickle("all_data.pickle")
    return df


def tokenize_doc(statement):
    """ Tokenize a document and return its bag-of-words representation as a dictionary """
    c = defaultdict(float)
    for word in statement:
        c[word] += 1
    return c


class NaiveBayes():
    """A Naive Bayes model for text classification"""
    def __init__(self, statements, labels, alpha):
        self.vocab = Counter([word for content in statements for word in content])
        self.word_to_idx = {k: v+1 for v, k in enumerate(self.vocab)} # word to index mapping
        self.word_to_idx["UNK"] = 0 # all the unknown words will be mapped to index 0
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1, NONE_LABEL: 2}
        self.idx_to_label = [POS_LABEL, NEG_LABEL, NONE_LABEL]
        self.vocab = set(self.word_to_idx.keys())
        self.alpha = alpha
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0,
                                        NONE_LABEL: 0.0}
        
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0,
                                         NONE_LABEL: 0.0}
        
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float),
                                   NONE_LABEL: defaultdict(float)}
    
    
    def train_model(self, statements, labels):
        """Trains the model one statement at a time"""
        for i in range(len(statements)):
            self.tokenize_and_update_model(statements[i], labels[i])
    
    def update_model(self, bow, label):
        """Update the NB model"""
        for word in bow.keys():
            self.class_word_counts[label][word] += bow[word]
        self.class_total_word_counts[label] += sum(bow.values())
        self.class_total_doc_counts[label] += 1
        return 
    
    def tokenize_and_update_model(self, doc, label):
        """Tokenizes a document doc and updates internal count statistics."""
        
        bow = tokenize_doc(doc)
        self.update_model(bow, label)
    
    def p_word_given_label(self, word, label):
        """Returns the probability of word given label (i.e., P(word|label))"""
        count_in_label= self.class_word_counts[label][word]
        all_in_label = self.class_total_word_counts[label]
        return (count_in_label/all_in_label)
    
    def p_word_given_label_and_psuedocount(self, word, label):
        """Returns the probability of word given label wrt psuedo counts."""
        count_in_label= self.class_word_counts[label][word]
        all_in_label = self.class_total_word_counts[label]
        return (count_in_label + self.alpha)/(all_in_label + (len(self.vocab)*self.alpha))
    
    def log_likelihood(self, bow, label):
        """Computes the log likelihood of a set of words give a label and psuedocount. """
        log_probs = [np.log(self.p_word_given_label_and_psuedocount(word, label)) for word in bow.keys()]
        return sum(log_probs)
    
    def log_prior(self, label):
        """Returns the fraction of training documents that are of class 'label'."""
        prior = self.class_total_doc_counts[label] / sum(self.class_total_doc_counts.values())
        return np.log(prior)
    
    def unnormalized_log_posterior(self, bow, label):
        """Computes the unnormalized log posterior (of doc being of class 'label')."""
        return self.log_prior(label) + self.log_likelihood(bow, label)
    
    def classify(self, bow):
        """Make classification of the document given a bow"""
        pos_log_post = self.unnormalized_log_posterior(bow, POS_LABEL)
        neg_log_post = self.unnormalized_log_posterior(bow, NEG_LABEL)
        none_log_post = self.unnormalized_log_posterior(bow, NONE_LABEL)
        if np.max([pos_log_post,neg_log_post,none_log_post]) == pos_log_post:
            return POS_LABEL
        elif np.max([pos_log_post,neg_log_post,none_log_post]) == neg_log_post:
            return NEG_LABEL
        else:
            return NONE_LABEL
    
    def likelihood_log_ratio(self, word, label):
        """Returns the ratio of P(word|label ) to P(word|not label) """
        if label == POS_LABEL:
            ratio = self.p_word_given_label_and_psuedocount(word, POS_LABEL) / ((self.p_word_given_label_and_psuedocount(word, NONE_LABEL) + self.p_word_given_label_and_psuedocount(word, NEG_LABEL)))
        elif label == NEG_LABEL:
            ratio = self.p_word_given_label_and_psuedocount(word, NEG_LABEL) / ((self.p_word_given_label_and_psuedocount(word, POS_LABEL) + self.p_word_given_label_and_psuedocount(word, NONE_LABEL)))
        else:
            ratio = self.p_word_given_label_and_psuedocount(word, NONE_LABEL) / ((self.p_word_given_label_and_psuedocount(word, NEG_LABEL) + self.p_word_given_label_and_psuedocount(word, POS_LABEL)))
        return np.log(ratio)
    
    def evaluate_classifier_accuracy(self, statements, labels):
        """Classify all statements in test data and return the accuracy"""
        accuracy = []
        for i in range(len(statements)):
            label = labels[i]
            bow = tokenize_doc(statements[i])
            predicted = self.classify(bow)
            if predicted == label:
                accuracy.append(1.0)
            else:
                accuracy.append(0.0)
        return sum(accuracy)/float(len(accuracy))


def make_train_test_data(df, label, test_size = 0.2):
    """Divide data into train and test data for model training and validating."""
    X = df.statements
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)#, random_state=42)
    return X_train, X_test, y_train, y_test



def param_fitting(train_data,train_labels):
    """
    This function takes the training data/labels and returns best performing alpha parameter.
    Use 5 fold cross validation to hypertune the parameter.
    """
    #x = np.arange(.05,2.05,.05)
    x = [.0001,.0005, .001, .005, .01, .05, .1, .5, 1, 5, 10,50,100]
    scores = []
    kf = KFold(n_splits = 5)
    for alpha in x:
        kfold_scores = []
        for train, test in kf.split(train_data):
            nb_model = NaiveBayes(train_data[train],train_labels[train], alpha)
            nb_model.train_model(train_data[train], train_labels[train])
            kfold_scores.append(nb_model.evaluate_classifier_accuracy(train_data[test], train_labels[test]))
        scores.append(np.mean(kfold_scores))      
    return x[np.argmax(scores)]    


def print_top_words(ll1, ll2, ll3, num = 5):
    """ Print the words with the highest log likelihood ratios """
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    for label in labels:
        pos = sorted(ll1[label].items(), key=lambda t: t[1], reverse = True)
        neg = sorted(ll2[label].items(), key=lambda t: t[1], reverse = True)
        none = sorted(ll3[label].items(), key=lambda t: t[1], reverse = True)
        print(f'Top {num} positive words for {label} =')
        print([i[0] for i in pos[0:num]])
        print(f'Top {num} negative words for {label} =')
        print([i[0] for i in neg[0:num]])
        print(f'Top {num} neutral words for {label} = ')
        print([i[0] for i in none[0:num]])        
        print()



def main(df, lik_log_rats_pos, lik_log_rats_neg, lik_log_rats_none):
    """ Driver function that trains a model for each test. Takes in the dictionaries
    that stores the log likelihood ratios for each word in order to later see which
    are the most impactful words.
    """
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    scores = []
    #store log likelihoods for each label
    for label in labels:
        train_data, test_data, train_labels, test_labels = make_train_test_data(df, label) 
        #alpha = param_fitting(train_data, train_labels)
        alpha = 1
        nb_model = NaiveBayes(train_data,train_labels, alpha)
        nb_model.train_model(train_data, train_labels)
        score = nb_model.evaluate_classifier_accuracy(test_data, test_labels)
        print(f"Score for {label} = {score}")
        scores.append(score)
        #update the log likelihood dictionaries
        for word in nb_model.vocab:
            lik_log_rats_pos[label][word] += nb_model.likelihood_log_ratio(word,POS_LABEL)
            lik_log_rats_neg[label][word] += nb_model.likelihood_log_ratio(word,NEG_LABEL)
            lik_log_rats_none[label][word] += nb_model.likelihood_log_ratio(word,NONE_LABEL)
        #print(nb_model.likelihood_log_ratio('bias',POS_LABEL))
        #print(lik_log_rats_pos[label]['bias'])    
    return scores   


if __name__ == '__main__':
    """ Due to the random component and lack of data we want to run it multiple
    times and average the scores.
    """
    NUM_EPOCHS = 1000
    all_scores = []
    df = load_data()
    #create dictionaries to store the log likelihoods for each word
    lik_log_rats_pos = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    lik_log_rats_neg = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    lik_log_rats_none = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    for i in range(NUM_EPOCHS):
        print(i)
        scores = main(df, lik_log_rats_pos, lik_log_rats_neg, lik_log_rats_none)
        all_scores.append(scores)
    all_scores = np.array(all_scores)    
    print()
    print("Average score for each independent variable = ")
    print(np.mean(all_scores,axis= 0))
    print()
    print_top_words(lik_log_rats_pos,lik_log_rats_neg,lik_log_rats_none)
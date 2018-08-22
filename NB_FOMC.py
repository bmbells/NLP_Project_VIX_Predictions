from collections import Counter, defaultdict
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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


def make_train_test_data(df, label):#, test_size = 0.2):
    """Divide data into train and test data for model training and validating."""
    df_train = df[df.set == "train"]
    df_test = df[df.set == "test"]
    X_train = df_train.statements
    X_test = df_test.statements
    y_train = df_train[label]
    y_test = df_test[label]
    return X_train, X_test, y_train, y_test


def param_fitting(train_data,train_labels):
    """
    This function takes the training data/labels and returns best performing alpha parameter.
    Use 5 fold cross validation to hypertune the parameter.
    """
    x = [.0001,.001,.01,.1,1]
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


def main():
    """ Driver function that trains a model for each test. It also updates our 
    log likelihood dictionaries to examine which words are most impactful.
    """
    labels = ['vix_buckets_1d', 'vix_buckets_5d', 'tnx_buckets_1d', 'tnx_buckets_5d']
    scores = []
    #store log likelihoods for each label
    df = load_data()
    #create log likelihood_dictionaries
    lik_log_rats_pos = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    lik_log_rats_neg = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    lik_log_rats_none = {'vix_buckets_1d' : defaultdict(float), 'vix_buckets_5d': defaultdict(float),
                    'tnx_buckets_1d' : defaultdict(float), 'tnx_buckets_5d': defaultdict(float)}
    for label in labels:
        train_data, test_data, train_labels, test_labels = make_train_test_data(df, label) 
        alpha = param_fitting(train_data, train_labels)
        print(f"Alpha for {label} = {alpha}")
        nb_model = NaiveBayes(train_data,train_labels, alpha)
        nb_model.train_model(train_data, train_labels)
        score = nb_model.evaluate_classifier_accuracy(test_data, test_labels)
        print(f"Score for {label} = {score}")
        scores.append(score)
        for word in nb_model.vocab:
            lik_log_rats_pos[label][word] += nb_model.likelihood_log_ratio(word,POS_LABEL)
            lik_log_rats_neg[label][word] += nb_model.likelihood_log_ratio(word,NEG_LABEL)
            lik_log_rats_none[label][word] += nb_model.likelihood_log_ratio(word,NONE_LABEL)
    print()        
    print_top_words(lik_log_rats_pos,lik_log_rats_neg,lik_log_rats_none)     
    return scores   


if __name__ == '__main__':
    scores = main()
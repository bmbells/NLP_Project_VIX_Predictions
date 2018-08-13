import pandas as pd
import os
import numpy as np
import string
import nltk
from collections import Counter
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.collocations import *
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
os.chdir("C:\\Users\\jooho\\NLPProject")
#os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

def read_and_clean_df():
    df = pd.read_pickle("df_minutes.pickle")
    for i in range(len(df)):
        temp = df.iloc[i,0].split('\n\nShare\n\n')
        df.iloc[i,0] = temp[len(temp) - 1]
        temp2 = df.iloc[i,0].split('For immediate release')
        df.iloc[i,0] = temp2[len(temp2) - 1].replace('\n',' ').replace('\t', ' ').replace('\r',' ').replace('\xa0',' ').strip()
        for j in range(1994,2019):
            splitter = str(j) + ' Monetary policy'
            temp3 = df.iloc[i,0].split(splitter)
            df.iloc[i,0] = temp3[0]
        temp4 = df.iloc[i,0].split('Last Update:')
        df.iloc[i,0] = temp4[0]
        temp5 = df.iloc[i,0].split('Board of Governors of the Federal Reserve System')
        df.iloc[i,0] = temp5[0]
        temp6 = df.iloc[i,0].split('Implementation Note issued')
        df.iloc[i,0] = temp6[0]
        temp7 = df.iloc[i,0].split('Statement Regarding Purchases of')
        df.iloc[i,0] = temp7[0]
        temp8 = df.iloc[i,0].split('Maturity Extension Program and Reinvestment Policy')
        df.iloc[i,0] = temp8[0]
        temp9 = df.iloc[i,0].split('1. The Open Market Desk will issue a technical note shortly after')
        df.iloc[i,0] = temp9[0]
        temp10 = df.iloc[i,0].split('Statement from Federal Reserve Bank of New')
        df.iloc[i,0] = temp10[0]
    df = df.loc[df.index >'1999-01-01 00:00:00'] #Filter out the stuff before 1999
    bad_dates = ['2007-08-10','2007-08-17','2008-01-22','2008-03-11','2008-10-08','2010-05-09']
    df = df.loc[~df.index.isin(bad_dates)]
    return df

def tokenize_and_preprocess_bystatement(stng): #Preprocesses by each statement
    """
    here we need to decide how to split up the words and what words to get rid of.
    Are locations important?
    """
    neg_tokens, sentences = add_negation_remove_propnouns(stng) #Adds negation, removes Proper Nouns
    translator = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(translator) for w in neg_tokens] #Removes remaining punctuation
    sentences = [[w.translate(translator) for w in sentence] for sentence in sentences]
    words = [word for word in stripped if word.isalpha()] #Removes non-alphabetic words
    sentences = [[word for word in sentence if word.isalpha()] for sentence in sentences]
    print (sentences)
    stop_words = set(stopwords.words('english'))
    stop_words_withneg = []
    for i in stop_words: #Add stopwords and negation of stopwords
        stop_words_withneg.append(i)
        stop_words_withneg.append('not'+i)
    words = [w for w in words if not w in stop_words_withneg] #Remove stopwords
    sentences = [[w for w in sentence if not w in stop_words_withneg] for sentence in sentences]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words] #Stem words
    sentences = [[porter.stem(word) for word in sentence] for sentence in sentences]
    return stemmed, sentences

def add_negation_remove_propnouns(stng):
    words = stng.split()
    sentences = tokenizer.tokenize(stng)
    sentences = [item.split() for item in sentences]
    tagged_sent = pos_tag(words) #Tag Words
    propernouns = [word for word, pos in tagged_sent if pos=='NNP'] #Isolate Proper Nouns
    words = [w for w in words if not w in propernouns] #Remove all propernouns
    sentences = [[w for w in sentence if not w in propernouns] for sentence in sentences]
    negation = False
    delims = '?.,!;j'
    result = []
    for word in words: #Adds not in front of negated words
        stripped = word.strip(delims).lower() #Convert to lowercase
        negated = 'not'+ stripped if negation else stripped
        result.append(negated)
        if any(neg in word for neg in ['not',"n't",'no']):
            negation = not negation
        if any(c in word for c in delims):
            negation = False
    resultsent = []
    for sentence in sentences:
        result = []
        for word in sentence:
            stripped = word.strip(delims).lower() #Convert to lowercase
            negated = 'not'+ stripped if negation else stripped
            result.append(negated)
            if any(neg in word for neg in ['not',"n't",'no']):
                negation = not negation
            if any(c in word for c in delims):
                negation = False
        resultsent.append(result)
    return result, resultsent

def preprocess_total(df):
    alltokens = []
    alltokens = [tk for row in df.statements for tk in row] #Get tokens from every statement
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(alltokens) #Find bigrams
    finder.apply_freq_filter(5) #Need frequency of at least 5
    top_bigrams = finder.nbest(bigram_measures.pmi,50) #Find top 50 bigrams
    infreq = [k for k,v in Counter(alltokens).items() if v<5] #Find words that only appeared less than 5 times to eventually remove
    return top_bigrams, infreq

def preprocess_final(tokens, bigrams, infreq_words):
    sentence = " ".join(tokens) #Turn back into a string so I can replace bi_grams into 1 word feature
    for b1, b2 in bigrams:
        sentence = sentence.replace("%s %s" % (b1 ,b2), "%s%s" % (b1, b2))
    words = sentence.split()
    words = [w for w in words if not w in infreq_words] #Remove infrequent words
    return words

def preprocess_final_sentences(sentences, bigrams, infreq_words):
    finallst = []
    for lsts in sentences:
        sentence = " ".join(lst) #Turn back into a string so I can replace bi_grams into 1 word feature
        for b1, b2 in bigrams:
            sentence = sentence.replace("%s %s" % (b1 ,b2), "%s%s" % (b1, b2))
        words = sentence.split()
        words = [w for w in words if not w in infreq_words] #Remove infrequent words
        finallst.append(words)
    return finallst

def combine_with_financial_data(df):
    df2 = pd.read_csv("financial_data.csv", index_col = 0)
    df3 = df.join(df2, how = 'left')
    df4 = df3[['statements', 'vix_1d', 'tnx_1d', 'vix_5d', 'tnx_5d']]
    return df4

def make_buckets(df):
    df_new = df.copy()
    df_new['vix_buckets_1d'] = pd.cut(df_new.vix_1d, [-.212, -.0406, .01, .424] , labels = [-1,0,1])
    df_new['vix_buckets_5d'] = pd.qcut(df_new.vix_5d, 3 , labels = [-1,0,1])
    df_new['tnx_buckets_1d'] = pd.qcut(df_new.tnx_1d, 3 , labels = [-1,0,1])
    df_new['tnx_buckets_5d'] = pd.qcut(df_new.tnx_5d, 3 , labels = [-1,0,1])
    return df_new

if __name__ == '__main__':
    df = read_and_clean_df()
    df['statements'], df['sentences'] = df.statements.apply(tokenize_and_preprocess_bystatement)
    bigrams, infreq_words = preprocess_total(df)
    df['statements'] = df.statements.apply(preprocess_final, args = (bigrams, infreq_words))
    df['sentences'] = df.sentences.apply(preprocess_final_sentences, args = (bigrams, infreq_words))
    df2 = combine_with_financial_data(df)
    df3 = make_buckets(df2)
    df3.to_pickle("./all_data.pickle")

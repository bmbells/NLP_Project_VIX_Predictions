from gensim.models import Word2Vec
import pandas as pd
import os
os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

def load_data():
    df = pd.read_pickle("all_data.pickle")
    return df

#load data and concatonate all sentences into a list of sentences
df = load_data()
all_sentences = []
for i in range(len(df)):
    for j in range(len(df.sentences[i])):
        all_sentences.append(df.sentences[i][j])
    
model = Word2Vec(all_sentences, min_count = 0, window=5) #unsure best window 
model.train(all_sentences, total_examples=len(all_sentences), epochs = 10)
model.wv.most_similar(positive = 'high') 
model.wv['high'] #this is the vector we need

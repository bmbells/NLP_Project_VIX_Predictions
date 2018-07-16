import pandas as pd
import os
import numpy as np
os.chdir("C:\\Users\\dabel\\Documents\\Natural_Language_Processing_MPCS\\project")

def read_and_clean_df():
    df = pd.read_pickle("df_minutes.pickle")
    for i in range(len(df)):
        temp = df.iloc[i,0].split('\n\nShare\n\n')
        df.iloc[i,0] = temp[len(temp) - 1]    
        temp2 = df.iloc[i,0].split('For immediate release')
        df.iloc[i,0] = temp2[len(temp2) - 1].replace('\n',' ').replace('\t', ' ').replace('\r',' ')
    return df

def tokenize_and_preprocess(df):
    """
    here we need to decide how to split up the words and what words to get rid of.
    Are locations important?        
    """
    pass

def combine_with_financial_data(df):
    df2 = pd.read_csv("financial_data.csv", index_col = 0)
    df3 = df.join(df2, how = 'left')
    df4 = df3[['statements', 'vix_1d', 'tnx_1d', 'vix_5d', 'tnx_5d']]
    return df4

df = read_and_clean_df()
df2 = combine_with_financial_data(df)

# Predicting VIX moves after a Federal Open Market Committee Minutes Release
## Bryan Beller and Jooho Yu, August 2018

scrape_FOMC.py: Scrapes the web for all FOMC statements and stores them as a pandas dataframe. The resulting data is then stored in "df_minutes.pickle".

get_financial_data.R: Downloads daily price data for VIX and TNX. Calculates 1 and 5 day percent changes. Stores the data in "financial_data.csv".

statements_clean.py: Performs all necessary preprocessing of the statements. Combines financial data with the preprocessed statements. Divides data into training and testing sets. Stores the data in "all_data.pickle".

NB_FOMC.py: Trains and tests a Naive Bayes model

NN_FOMC.py: Trains and tests a Neural Network model

SVM_FOMC.py: Trains and tests a SVM model

RF_FOMC.py: Trains and tests a Random Forest model

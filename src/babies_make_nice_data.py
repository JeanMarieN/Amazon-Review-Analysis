# Script for reading a data set, remove NaN, tokenize it, split if needed
# write out again

# To read the data back into dataframes: 
"""
from ast import literal_eval

train = pd.read_csv('../../data/processed/tok_phase1_train80.csv')
validate = pd.read_csv('../../data/processed/tok_phase1_validate10.csv')
test = pd.read_csv('../../data/processed/tok_phase1_test10.csv')

for frame in [train, validate, test]:
    for col in ['summary_tokens', 'review_tokens']:
        frame[col] = frame[col].map(literal_eval)
"""


filename = '../../../data/raw/baby_reviews.csv'
fileout = 'baby_reviews_spell'
correct_spelling = True
split = False
split_train = 0.8

import pandas as pd 
import numpy as np 
import tknzr
from sklearn.model_selection import train_test_split
from nltk import TweetTokenizer

# read and shuffle data
df = pd.read_csv(filename)
np.random.seed(42)
# df = df.sample(frac=1).reset_index(drop=True) # Shuffle only if training set
print("Read {} rows.".format(len(df)))

# replace NaN values with empty strings
df.fillna('', inplace=True)

# create objects for processing strings
tokenizer = tknzr.make_tokenizer()
spell = tknzr.make_spellchecker()

# tokenize
df['review_tokens'] = ''
df['summary_tokens'] = ''
for i in range(len(df)):
    df.at[i, "review_tokens"] = tknzr.tokenize1(df.at[i, "reviewText"], correct_spelling, tokenizer, spell)
    df.at[i, 'summary_tokens'] = tknzr.tokenize1(df.at[i, "summary"], correct_spelling, tokenizer, spell)
print("Done tokenizing.")

# create token count columns 
df['summary_wc'] = df['summary_tokens'].map(len)
df['review_wc'] = df['review_tokens'].map(len)

# create standardised token count columns 
mean_summary_wc = np.mean(df['summary_wc'])
mean_review_wc = np.mean(df['review_wc'])
mean_year = np.mean(df['year'])

std_summary_wc = np.std(df['summary_wc'])
std_review_wc = np.std(df['review_wc'])
std_year = np.std(df['year'])

df['summary_wc_std'] = (df['summary_wc'] - mean_summary_wc) / std_summary_wc
df['review_wc_std'] = (df['review_wc'] - mean_review_wc) / std_review_wc
df['year_std'] = (df['year'] - mean_year) / std_year

# remove original strings
df.drop(['summary', 'reviewText'], axis=1, inplace=True)

# split if necessary
# write to file(s)
if split:
    train, test = train_test_split(df, test_size=1-split_train,random_state = 7)
    validate, test = train_test_split(test, test_size=0.5, random_state = 7)
    train.to_csv('../../../data/processed/tok_'+fileout+'_train'+str(int(round(split_train*100, 0)))+'.csv', index=False)
    validate.to_csv('../../../data/processed/tok_'+fileout+'_validate'+str(int(round((1-split_train)/0.02, 0)))+'.csv', index=False)
    test.to_csv('../../../data/processed/tok_'+fileout+'_test'+str(int(round((1-split_train)/0.02, 0)))+'.csv', index=False)
else:
    df.to_csv('../../../data/processed/tok_'+fileout+'.csv', index=False)

print("done with baby_reviews")


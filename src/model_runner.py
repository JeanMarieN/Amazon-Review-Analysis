import pandas as pd
from nltk.tokenize import TweetTokenizer
import re
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
# tensorflow and keras
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
# our libraries
import preprocess
import visualise



def tokenize(string):
    """
    takes string input and tokenizes into a list of strings. 
    If runtime is slow move tknzr out of function and call for it in the input. 
    This is highly unlikely given that function runs in O(1) = constant time.  
    """
    
    string = re.sub('\&quot;', '', string)
    
    tknzr = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
    tokens = tknzr.tokenize(string)
    return tokens


df = pd.read_csv('../data/raw/phase1_movie_reviews-train.csv')
df.fillna("", inplace = True)

padleft_int = 0
padleft_sym = '==padleft_sym=='
padright_int = 1
padright_sym = '==padright_sym=='
unknown_int = 2
unknown_sym = '==unknown_sym=='

print("Tokenizing "+str(len(df))+" strings.\nThis may take a while...")
df["reviewText"] = df["reviewText"].map(tokenize)
df["summary"] = df["summary"].map(tokenize)

vocab_counter = Counter()
for doc in list(df['summary']):
    vocab_counter.update(doc)
for doc in list(df['reviewText']):
    vocab_counter.update(doc) 

min_times_word_used = 5 # minimum 11 times for now - should get better with better tokens
print("\nDiscarding those tokens that appear less than {} times.".format(min_times_word_used))
print(len(vocab_counter), "tokens before.")
for key in list(vocab_counter.keys()):
    if vocab_counter[key] < min_times_word_used: 
        vocab_counter.pop(key)
print(len(vocab_counter), "tokens after.")    

vocab_set = set(vocab_counter.keys())

# vocabulary list and int map
vocab_list = ['==padleft_sym==', '==padright_sym==', '==unknown_sym=='] + sorted(vocab_set)
vocab_map = {word: i for i, word in enumerate(vocab_list)}

label_set = set(df['polarity'].unique())

# label list and int map
label_list = sorted(label_set)
label_map = {word: i for i, word in enumerate(label_list)}

print("Converting labels to binary encoding: ")
y = preprocess.create_one_hot(df['polarity'], label_map)

train_summary = preprocess.word_index(df['summary'], vocab_map, unknown_int)
train_review = preprocess.word_index(df['reviewText'], vocab_map, unknown_int) 

summary_len = max(map(len, list(df['summary'])))
review_len = 500

train_summary = sequence.pad_sequences(sequences=train_summary, 
                                        maxlen=summary_len, 
                                        dtype='int32', 
                                        padding='pre', 
                                        value=padleft_int)
train_review = sequence.pad_sequences(sequences=train_review, 
                                        maxlen=review_len, 
                                        dtype='int32', 
                                        padding='pre',
                                        truncating='pre',
                                        value=padleft_int)

embedding_dim = 300

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((len(vocab_list), embedding_dim))
for i, word in enumerate(vocab_list):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        pass #maybe use fuzzywuzzy

embedding_vector_length = embedding_dim
model = Sequential()
model.add(Embedding(len(vocab_list), embedding_vector_length, 
                    input_length=summary_len, weights=[embedding_matrix], trainable=True))
model.add(GRU(100, activation='relu', recurrent_activation='sigmoid', dropout=0.3, 
              recurrent_dropout=0.3, kernel_constraint=maxnorm(4), recurrent_constraint=maxnorm(5),
              # anything below here are default values 
              use_bias=True, kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', bias_initializer='zeros', 
              kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, 
              activity_regularizer=None,  
              bias_constraint=None, implementation=1, return_sequences=False, return_state=False, 
              go_backwards=False, stateful=False, unroll=False, reset_after=False))
model.add(Dense(len(label_set), activation='softmax'))
nadam = keras.optimizers.nadam(lr=0.0006)
model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])
print("\n", model.summary())
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
hist = model.fit(train_summary, y, validation_split=0.1, epochs=50, batch_size=128, callbacks=[es])


visualise.plot_results(hist.history['loss'], hist.history['acc'], 
                       "Training history", '../reports/figures/GRU_summary_training_hist.svg')
visualise.plot_results(hist.history['val_loss'], hist.history['val_acc'], 
                       "Validation history", '../reports/figures/GRU_summary_validation_hist.svg')

# Predict for training data 
#y_pred = model.predict(train_summary)

# Undo one-hot
#y_pred = preprocess.undo_one_hot(y_pred, label_list)
#y_orig = df['polarity']

#print("Splitting into train/test/validation...")
#train, test = train_test_split(df, test_size=0.2,random_state = 7)
#validation, test = train_test_split(test, test_size=0.5, random_state = 7)





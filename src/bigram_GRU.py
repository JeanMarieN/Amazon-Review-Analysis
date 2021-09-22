# external libraries
import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval
# tensorflow and keras
import keras.optimizers
from keras.datasets import imdb
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate 
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
# fix random seed for reproducibility - only works for CPU version of tensorflow
np.random.seed(42)
# our libraries
import preprocess
import visualise

train = pd.read_csv('../data/processed/tok_phase1_train80.csv')
validate = pd.read_csv('../data/processed/tok_phase1_validate10.csv')
test = pd.read_csv('../data/processed/tok_phase1_test10.csv')

print("\nFiles read, converting tokens to lists.")
for frame in [train, validate, test]:
    for col in ['summary_tokens', 'review_tokens']:
        frame[col] = frame[col].map(literal_eval)

### Preprocessing 
# declare the padding and unknown symbols
pad_mask_int = 0
pad_mask_sym = '==pad_mask=='
unknown_int = 1
unknown_sym = '==unknown_sym=='

# vocabulary set
vocab_counter = Counter()
for doc in train['summary_tokens']:
    vocab_counter.update(doc)
for doc in train['review_tokens']:
    vocab_counter.update(doc)    

min_times_word_used = 2 # if at least 2 then the model will be prepared for unknown words in test and validation sets
print(len(vocab_counter), "tokens before discarding those that appear less than {} times.".format(min_times_word_used))
for key in list(vocab_counter.keys()):
    if vocab_counter[key] < min_times_word_used: 
        vocab_counter.pop(key)
print(len(vocab_counter), "tokens after discarding those that appear less than {} times.".format(min_times_word_used))   
vocab_set = set(vocab_counter.keys())

# vocabulary list and int map
vocab_list = [pad_mask_sym, unknown_sym] + sorted(vocab_set)
vocab_map = {word: index for index, word in enumerate(vocab_list)}

# label set
label_set = set(train['polarity'].unique())

# label list and int map
label_list = sorted(label_set)
label_map = {word: index for index, word in enumerate(label_list)}

# create one-hot sparse matrix of labels
y_train = preprocess.create_one_hot(train['polarity'], label_map)
y_validate = preprocess.create_one_hot(validate['polarity'], label_map)
y_test = preprocess.create_one_hot(test['polarity'], label_map)

# replace strings with ints (tokenization is done on the Series fed to word_index())
train_summary = preprocess.word_index(train['summary_tokens'], vocab_map, unknown_int)
train_review = preprocess.word_index(train['review_tokens'], vocab_map, unknown_int) 

validate_summary = preprocess.word_index(validate['summary_tokens'], vocab_map, unknown_int)
validate_review = preprocess.word_index(validate['review_tokens'], vocab_map, unknown_int) 

test_summary = preprocess.word_index(test['summary_tokens'], vocab_map, unknown_int)
test_review = preprocess.word_index(test['review_tokens'], vocab_map, unknown_int) 

# pad / truncate 
from keras.preprocessing.sequence import pad_sequences

summary_len = max(map(len, list(train['summary_tokens'])))
review_len = 500

train_summary = pad_sequences(sequences=train_summary, 
                              maxlen=summary_len, 
                              dtype='int32', 
                              padding='pre', 
                              value=pad_mask_int)
train_review = pad_sequences(sequences=train_review, 
                             maxlen=review_len, 
                             dtype='int32', 
                             padding='pre',
                             truncating='pre',
                             value=pad_mask_int)

validate_summary = pad_sequences(sequences=validate_summary, 
                              maxlen=summary_len, 
                              dtype='int32', 
                              padding='pre', 
                              value=pad_mask_int)
validate_review = pad_sequences(sequences=validate_review, 
                             maxlen=review_len, 
                             dtype='int32', 
                             padding='pre',
                             truncating='pre',
                             value=pad_mask_int)

test_summary = pad_sequences(sequences=test_summary, 
                              maxlen=summary_len, 
                              dtype='int32', 
                              padding='pre', 
                              value=pad_mask_int)
test_review = pad_sequences(sequences=test_review, 
                             maxlen=review_len, 
                             dtype='int32', 
                             padding='pre',
                             truncating='pre',
                             value=pad_mask_int)

# pretrained embeddings are from https://nlp.stanford.edu/projects/glove/
# start by loading in the embedding matrix
# load the whole embedding into memory
print("\nReading big ol' word embeddings")
embeddings_index = dict()
with open('../data/external/glove.42B.300d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_dim = 300

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((len(vocab_list), embedding_dim))
for i, word in enumerate(vocab_list):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # print(word)
        pass # maybe we should use fuzzywuzzy to get vector of nearest word? Instead of all zeros

del embeddings_index

# Keras functional API for joined model
input_s1 = Input(shape=(summary_len-1,), dtype='int32', name='input_s1')
input_r1 = Input(shape=(review_len-1,), dtype='int32', name='input_r1')
input_s2 = Input(shape=(summary_len-1,), dtype='int32', name='input_s2')
input_r2 = Input(shape=(review_len-1,), dtype='int32', name='input_r2')

embedding_vector_length = embedding_dim
GRU_nodes_summary = 100
GRU_nodes_review = 150

emb = Embedding(len(vocab_list), embedding_vector_length, mask_zero=True,
                weights=[embedding_matrix], trainable=False)

emb_s1 = emb(input_s1)
emb_r1 = emb(input_r1)
emb_s2 = emb(input_s2)
emb_r2 = emb(input_r2)

emb_s = Concatenate()([emb_s1, emb_s2])
emb_r = Concatenate()([emb_r1, emb_r2])

gru_s = GRU(GRU_nodes_summary, activation='tanh', recurrent_activation='sigmoid', dropout=0.3, 
              recurrent_dropout=0.4, kernel_constraint=maxnorm(4), recurrent_constraint=maxnorm(5),
              unroll=True, 
            
              use_bias=True, kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', bias_initializer='zeros', 
              kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, 
              activity_regularizer=None,  
              bias_constraint=None, implementation=1, return_sequences=False, return_state=False, 
              go_backwards=False, stateful=False, reset_after=False)(emb_s)
gru_r = GRU(GRU_nodes_review, activation='tanh', recurrent_activation='sigmoid', dropout=0.3, 
              recurrent_dropout=0.4, unroll=True, 
              
              kernel_constraint=None, recurrent_constraint=None,
              use_bias=True, kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', bias_initializer='zeros', 
              kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, 
              activity_regularizer=None,  
              bias_constraint=None, implementation=1, return_sequences=False, return_state=False, 
              go_backwards=False, stateful=False, reset_after=False)(emb_r)

concat = Concatenate()([gru_s, gru_r])
output = Dense(len(label_set), activation='softmax')(concat)
model = Model([input_s1, input_s2, input_r1, input_r2], output)
nadam1 = keras.optimizers.nadam(lr=0.0006)
model.compile(loss='categorical_crossentropy', optimizer=nadam1, metrics=['accuracy'])

# unfrozen embeddings
emb.trainable = True
thawn = Model([input_s1, input_s2, input_r1, input_r2], output)
nadam2 = keras.optimizers.nadam(lr=0.00006)
thawn.compile(loss='categorical_crossentropy', optimizer=nadam2, metrics=['accuracy'])

print(model.summary())
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
hist1 = model.fit(x=[train_summary[:, :-1], train_summary[:, 1:], 
                     train_review[:, :-1], train_review[:, 1:]], 
                  y=y_train, 
                  validation_data=([validate_summary[:, :-1], validate_summary[:, 1:], 
                                    validate_review[:, :-1], validate_review[:, 1:]], 
                                   y_validate), 
                  epochs=50, batch_size=128, callbacks=[es])

hist2 = thawn.fit(x=[train_summary[:, :-1], train_summary[:, 1:], 
                     train_review[:, :-1], train_review[:, 1:]], 
                  y=y_train, 
                  validation_data=([validate_summary[:, :-1], validate_summary[:, 1:], 
                                    validate_review[:, :-1], validate_review[:, 1:]], 
                                   y_validate), 
                  epochs=50, batch_size=128, callbacks=[es])


visualise.plot_both_results(hist1.history['loss'] + hist2.history['loss'],
                       hist1.history['acc'] + hist2.history['acc'], 
                       hist1.history['val_loss'] + hist2.history['val_loss'],
                       hist1.history['val_acc'] + hist2.history['val_acc'], 
                       "History", '../reports/figures/GRU_summary_final_hist.svg')

print("Training score")
print(thawn.evaluate([train_summary[:, :-1], train_summary[:, 1:], train_review[:, :-1], train_review[:, 1:]], y_train))

print("Validation score")
print(thawn.evaluate([validate_summary[:, :-1], validate_summary[:, 1:], validate_review[:, :-1], validate_review[:, 1:]], y_validate))

# Predict for validation data 
# y_pred = model.predict([validate_summary, validate_review])

# Undo one-hot
# y_pred = preprocess.undo_one_hot(y_pred, label_list)
# y_orig = validate['polarity']

# visualise.plot_confusion(y_orig, y_pred, label_list) # yeah, need to fix so figure is saved instead

print("\nChecking for weights that have gone NaN (that's a bad thing):")
for weight in model.get_weights():
    df = pd.DataFrame(weight)
    print(df[df.isnull().any(axis=1)])

print("\nDone")

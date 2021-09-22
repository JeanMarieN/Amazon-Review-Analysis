# external libraries
import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval
import time
import sys 
from shutil import copyfile
import json
# tensorflow and keras
import keras.optimizers
from keras.datasets import imdb
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Bidirectional, Dropout
from keras.layers import GRU, CuDNNGRU, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import L1L2
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
# fix random seed for reproducibility - only works for CPU version of tensorflow
np.random.seed(42)
# our libraries
import preprocess
import visualise

train = pd.read_csv('../../../data/processed/tok_phase1_movie_reviews-train_train80.csv')
validate = pd.read_csv('../../../data/processed/tok_phase1_movie_reviews-train_validate10.csv')
test = pd.read_csv('../../../data/processed/tok_phase1_movie_reviews-train_test10.csv')

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
train_summary = preprocess.word_index(train['summary_tokens'], vocab_map, unknown_int, reverse=False)
train_review = preprocess.word_index(train['review_tokens'], vocab_map, unknown_int, reverse=False) 

validate_summary = preprocess.word_index(validate['summary_tokens'], vocab_map, unknown_int, reverse=False)
validate_review = preprocess.word_index(validate['review_tokens'], vocab_map, unknown_int, reverse=False) 

test_summary = preprocess.word_index(test['summary_tokens'], vocab_map, unknown_int, reverse=False)
test_review = preprocess.word_index(test['review_tokens'], vocab_map, unknown_int, reverse=False) 

# pad / truncate 
from keras.preprocessing.sequence import pad_sequences

summary_len = max(map(len, list(train['summary_tokens'])))
review_len = 300

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
embeddings_index_1 = dict()
with open('../../../data/external/glove.42B.300d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print(values)
        embeddings_index_1[word] = coefs
print('Loaded %s word vectors.' % len(embeddings_index_1))

#embeddings_index_2 = dict()
#with open('../../../data/external/sentiment_dic.json') as f:
#    embeddings_index_2 = json.load(f)
#for key, value in embeddings_index_2.items():
#    embeddings_index_2[key] = np.array([value])
#print('Loaded %s word vectors.' % len(embeddings_index_2))

embedding_dim_1 = 300
embedding_dim_2 = 0

embedding_dim = embedding_dim_1 + embedding_dim_2

# create a weight matrix for words in training docs
if embedding_dim_2 > 0:
    embedding_matrix = np.zeros((len(vocab_list), embedding_dim))
    for i, word in enumerate(vocab_list):
        embedding_vector_1 = embeddings_index_1.get(word)
        embedding_vector_2 = embeddings_index_2.get(word)
        if embedding_vector_1 is not None and embedding_vector_2 is not None:
            embedding_matrix[i] = np.concatenate((embedding_vector_1, embedding_vector_2))
        elif embedding_vector_1 is None and embedding_vector_2 is not None:
            embedding_matrix[i] = np.concatenate((np.zeros(embedding_dim_1), embedding_vector_2))
        elif embedding_vector_1 is not None and embedding_vector_2 is None:
            embedding_matrix[i] = np.concatenate((embedding_vector_1, np.zeros(embedding_dim_2)))
        else:
            # print(word)
            pass # maybe we should use fuzzywuzzy to get vector of nearest word? Instead of all zeros
else:
    embedding_matrix = np.zeros((len(vocab_list), embedding_dim))
    for i, word in enumerate(vocab_list):
        embedding_vector = embeddings_index_1.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # print(word)
            pass # maybe we should use fuzzywuzzy to get vector of nearest word? Instead of all zeros


# Name run for tensorboard
NAME = 'redropout-GRU-{}'.format(time.strftime('%y%m%d_%H%M', time.localtime(time.time())))

# Copy this python file into the TensorBoard logs folder
copyfile(sys.argv[0], './tb_logs/{}.py'.format(NAME))

# Define regularisers
#regularizers  = [L1L2(l1=0.0, l2=0.0)]
#regularizers += [L1L2(l1=0.000001, l2=0.0), L1L2(l1=0.0, l2=0.000001), L1L2(l1=0.000001, l2=0.000001)]
#regularizers += [L1L2(l1=0.0000001, l2=0.0), L1L2(l1=0.0, l2=0.0000001), L1L2(l1=0.0000001, l2=0.0000001)]
#regularizers += [L1L2(l1=0.00000001, l2=0.0), L1L2(l1=0.0, l2=0.00000001), L1L2(l1=0.00000001, l2=0.00000001)]

for size in [0.1, 0.3, 0.4, 0.5, 0.7]:
    # Keras functional API for joined model
    input_s = Input(shape=(summary_len,), dtype='int32', name='input_s')
    input_r = Input(shape=(review_len,), dtype='int32', name='input_r')

    embedding_vector_length = embedding_dim
    GRU_nodes_summary = 64
    GRU_nodes_review = 100

    emb = Embedding(len(vocab_list), embedding_vector_length, mask_zero=False,
                    weights=[embedding_matrix], trainable=False)

    emb_s = emb(input_s)
    emb_r = emb(input_r)

    gru_s = GRU(GRU_nodes_summary,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=L1L2(l1=0.1, l2=0.0),
        activity_regularizer=L1L2(l1=1e-07, l2=0.0),
        kernel_constraint=maxnorm(3),
        recurrent_constraint=maxnorm(3),
        bias_constraint=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        dropout=0.3,
        recurrent_dropout=size,
        unroll=True,
        activation='tanh',
        recurrent_activation='sigmoid')(emb_s)

    gru_r = GRU(GRU_nodes_review,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=L1L2(l1=0.1, l2=0.0),
        activity_regularizer=L1L2(l1=1e-07, l2=0.0),
        kernel_constraint=maxnorm(3),
        recurrent_constraint=maxnorm(3),
        bias_constraint=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        dropout=0.3,
        recurrent_dropout=size,
        unroll=True,
        activation='tanh',
        recurrent_activation='sigmoid')(emb_r)

    concat = Concatenate()([gru_s, gru_r])
    dropout = Dropout(rate=0.6)(concat)
    output = Dense(len(label_set), activation='softmax')(dropout)
    model = Model([input_s, input_r], output)
    nadam = keras.optimizers.nadam(lr=0.0003)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

    # unfrozen embeddings
#    emb.trainable = True
#    thawn = Model([input_s, input_r], output)
#    thawn.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

    print(model.summary())
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    tensorboard = TensorBoard(log_dir = './tb_logs/{}'.format(str(size)+'_'+NAME))

    hist1 = model.fit(x=[train_summary, train_review], 
                    y=y_train, 
                    validation_data=([validate_summary, validate_review], 
                                    y_validate), 
                    epochs=50, batch_size=64, callbacks=[es, tensorboard])

#    hist2 = thawn.fit(x=[train_summary, train_review], 
#                    y=y_train, 
#                    validation_data=([validate_summary, validate_review], 
#                                    y_validate), 
#                    epochs=50, batch_size=128, callbacks=[es, tensorboard])


### Everything below here should be available in Tensorboard? 

# visualise.plot_results(hist.history['loss'], hist.history['acc'], 
#                        "Training history", '../reports/figures/GRU_summary_training_hist.svg')
# visualise.plot_results(hist.history['val_loss'], hist.history['val_acc'], 
#                        "Validation history", '../reports/figures/GRU_summary_validation_hist.svg')

# Predict for validation data 
# y_pred = model.predict([validate_summary, validate_review])

# Undo one-hot
# y_pred = preprocess.undo_one_hot(y_pred, label_list)
# y_orig = validate['polarity']

# visualise.plot_confusion(y_orig, y_pred, label_list) # yeah, need to fix so figure is saved instead

# print("\nChecking for weights that have gone NaN (that's a bad thing):")
# for weight in model.get_weights():
    # df = pd.DataFrame(weight)
    # print(df[df.isnull().any(axis=1)])


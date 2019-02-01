# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:22:35 2019

@author: Amin
"""


from combiner import combiner  # to get the results of all text in the folder into a string
import numpy as np
import sys

# keras imports 
from keras.utils import np_utils  # to perform onehot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

corpus, _, med_length = combiner('test')  # recieveing the corpus
chars = sorted(list(set(corpus)))  # getting a list of unique chars
n_chars = len(corpus)  # number of characters in the document
n_vocab = len(chars)  # total number of characters
print("total characters :", n_chars)
print("total vocab :", n_vocab)
seq_length = 4 * int(med_length) # 4 sentence at a time

#  developing char to index and index to char 
char_to_ind = {char: i for i, char in enumerate(chars)}
ind_to_char = {i: char for i, char in enumerate(chars)}

# preparing data set of input output pairs
X = []
y= []
for i in range(0, n_chars - seq_length ):
    seq_in = corpus[i : i + seq_length]
    seq_out = corpus[i + seq_length]
    X.append([char_to_ind[c] for c in seq_in])
    y.append(char_to_ind[seq_out])
n_patterns = len(X)
print("total patterns: ", n_patterns)
dataX = X # to be used for text generation section
# reshaping X to fit keras input shape
# reshaping y to fit softmax output
X = np.reshape(X, (n_patterns, seq_length, 1))
X = X / float(n_vocab)  # normalizing over lenght of vocabulary
y = np_utils.to_categorical(y)  #onehot encoding

# developing a simple LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.5))  # originaly 0.2
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
#  define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list= [checkpoint]
model.fit(X, y, epochs=7, batch_size= 128, callbacks = callbacks_list)

# reloading weights
filename = "weights-improvement-02-2.7099.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# picking up a random seed
np.random.seed(26)
start = np.random.randint(0, len(dataX)-1)
pattern = (dataX[start])
print("seed:")
print("\"", "".join(ind_to_char[value] for value in pattern), "\"")

for i in range(2 * seq_length):
    x = np.reshape(pattern, (1,len(pattern),1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.random.choice(range(n_vocab), p = prediction.ravel()) 
    #index = np.argmax(prediction)
    result = ind_to_char[index]
    seq_in = [ind_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone")

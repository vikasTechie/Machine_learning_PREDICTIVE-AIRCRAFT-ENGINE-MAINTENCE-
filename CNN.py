# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uKtSs2-fvxHPQ6eN9adubCE9k5MuBrKf
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as pt
import random
import os

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *

cd /content/drive/My Drive/nasaDataset

ls

trainDataset = pd.read_csv('./train_FD004.txt', sep=" ", header=None)
trainDataset.drop(trainDataset.columns[[26,27]],axis=1,inplace=True)
trainDataset = trainDataset.head(10000)
trainDataset.columns = ['idNumber', 'cycleNumber', 'op1', 'op2', 'op3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
print('#id:',len(trainDataset.idNumber.unique()))
trainDataset = trainDataset.sort_values(['idNumber','cycleNumber'])
print(trainDataset.shape)
trainDataset.head(3)

pt.figure(figsize=(50,6))
trainDataset.idNumber.value_counts().plot.bar()
print("medium working time:", trainDataset.idNumber.value_counts().mean())
print("max working time:", trainDataset.idNumber.value_counts().max())
print("min working time:", trainDataset.idNumber.value_counts().min())

engine_idNumber = trainDataset[trainDataset['idNumber'] == 1]

ax = engine_idNumber[trainDataset.columns[2:]].plot(subplots=True, sharex=True, figsize=(20,30))

testDataset = pd.read_csv('./test_FD004.txt', sep=" ", header=None)
testDataset.drop(testDataset.columns[[26, 27]], axis=1, inplace=True)
testDataset.columns = ['idNumber', 'cycleNumber', 'op1', 'op2', 'op3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
print('#id:',len(testDataset.idNumber.unique()))
print(testDataset.shape)
testDataset.head(3)

GroundTruth = pd.read_csv('./RUL_FD004.txt', sep=" ", header=None)
GroundTruth.drop(GroundTruth.columns[[1]], axis=1, inplace=True)
GroundTruth.columns = ['RUL']
GroundTruth = GroundTruth.set_index(GroundTruth.index + 1)

print(GroundTruth.shape)
GroundTruth.head(3)

trainDataset['RUL']=trainDataset.groupby(['idNumber'])['cycleNumber'].transform(max)-trainDataset['cycleNumber']
trainDataset.RUL[0:10]

window1 = 50
window0 = 10
trainDataset['label1'] = np.where(trainDataset['RUL'] <= window1, 1, 0 )
trainDataset['label2'] = trainDataset['label1']
trainDataset.loc[trainDataset['RUL'] <= window0, 'label2'] = 2
trainDataset.head()

### SCALE TRAIN DATA ###

def scale(df):
    
    return (df - df.min())/(df.max()-df.min())

for col in trainDataset.columns:
    if col[0] == 's':
        trainDataset[col] = scale(trainDataset[col])

        
trainDataset = trainDataset.dropna(axis=1)
trainDataset.head()

### CALCULATE RUL TEST ###
GroundTruth['max'] = testDataset.groupby('idNumber')['cycleNumber'].max() + GroundTruth['RUL']
testDataset['RUL'] = [GroundTruth['max'][i] for i in testDataset.idNumber] - testDataset['cycleNumber']

testDataset['label1'] = np.where(testDataset['RUL'] <= window1, 1, 0 )
testDataset['label2'] = testDataset['label1']
testDataset.loc[testDataset['RUL'] <= window0, 'label2'] = 2

for col in testDataset.columns:
    if col[0] == 's':
        testDataset[col] = scale(testDataset[col])

        
testDataset = testDataset.dropna(axis=1)
testDataset.head()

sequence_length = 50

def gen_sequence(id_df, seq_length, seq_cols):
    #print("hellp")
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    #print(num_elements)
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 (start stop) -> from row 0 to row 50
    # 1 51 (start stop) -> from row 1 to row 51
    # 2 52 (start stop) -> from row 2 to row 52
    # ...
    # 141 191 (start stop) -> from row 141 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
    #print(data_matrix)    
def gen_labels(id_df, seq_length, label):

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

sequence_cols = []
for col in trainDataset.columns:
    if col[0] == 's' or col[0] =='o':
        sequence_cols.append(col)
#sequence_cols.append('cycle_norm')
print(sequence_cols)

x_train, x_test = [], []
for engine_id in trainDataset.idNumber.unique():
    for sequence in gen_sequence(trainDataset[trainDataset.idNumber==engine_id], sequence_length, sequence_cols):
        #print(engine_id)
        #print(trainDataset[trainDataset.idNumber==engine_id])
        #print(sequence)
        x_train.append(sequence)
        #break;
    for sequence in gen_sequence(testDataset[testDataset.idNumber==engine_id], sequence_length, sequence_cols):
        x_test.append(sequence)
    
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

print("X_Train shape:", x_train.shape)
print("X_Test shape:", x_test.shape)



y_train, y_test = [], []
for engine_id in trainDataset.idNumber.unique():
    for label in gen_labels(trainDataset[trainDataset.idNumber==engine_id], sequence_length, ['label2'] ):
        #print(label)
        y_train.append(label)
        
    for label in gen_labels(testDataset[testDataset.idNumber==engine_id], sequence_length, ['label2']):
        y_test.append(label)
    
y_train = np.asarray(y_train).reshape(-1,1)
y_test = np.asarray(y_test).reshape(-1,1)
print(y_train)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

y_train = to_categorical(y_train)
print(y_train.shape)

y_test = to_categorical(y_test)
print(y_test.shape)







def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

pt.figure(figsize=(20,20))
for i in range(0,17):
    
    pt.subplot(6, 3, i+1)    
    rec = rec_plot(x_train[0,:,i])
    pt.imshow(rec)
    pt.title(sequence_cols[i])
pt.show()

x_train_img = np.apply_along_axis(rec_plot, 1, x_train).astype('float16')
print(x_train_img.shape)

x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')
print(x_test_img.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 24)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())







tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=6)

model.fit(x_train_img, y_train, batch_size=512, epochs=25, callbacks=[es],validation_split=0.2, verbose=2)

model.evaluate(x_test_img, y_test, verbose=2)
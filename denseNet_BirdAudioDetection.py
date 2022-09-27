# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 06:18:23 2020

@author: rajes
"""

import librosa
import numpy as np
import csv
import glob
from scipy import io
import pandas as pd
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from scipy.fft import fft
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import TimeDistributed


# dirList = []
# for folder in ['ff1010bird_wav','warblrb10k_public_wav'] :
#     dirList+=(glob.glob("C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/train_data/"+folder+"/wav/*.wav"))

# ff1010bird_matrix = pd.read_csv('C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/train_data/ff1010bird_metadata_2018.csv',sep=',', engine='python')
# warblrb10k_public_matrix = pd.read_csv('C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/train_data/warblrb10k_public_metadata_2018.csv',sep=',', engine='python')
# total_matrix =ff1010bird_matrix.append(warblrb10k_public_matrix)

# X=[]
# Y=[]
# count = 0
# y = 0
# input_length = 441000
# for audio_path in dirList :
#     count+=1
#     print(count)
#     itemid = audio_path.split("/")[8].split("\\")[1].split('.')[0]
#     if itemid.isnumeric() :
#         itemid=int(itemid) 
#     y = int(total_matrix[total_matrix['itemid']==itemid].hasbird)
 
    
#     audio_data = librosa.load(audio_path,sr=44100)[0]
#     if len(audio_data) > input_length:
#             audio_data = audio_data[:input_length]
#     else:
#             audio_data = np.pad(audio_data, (0, max(0, input_length - len(audio_data))), "constant")
    
    
#     audio_data = audio_data * 1/np.max(np.abs(audio_data))
    
#     kwargs_for_mel = {'n_mels': 40,'fmax':22050}
#     x = librosa.feature.melspectrogram(
#         y=audio_data, 
#         sr=44100, 
#         n_fft=1024, 
#         hop_length=512, 
#         **kwargs_for_mel)
#     X.append(np.array(x))
#     Y.append(y)

# X_=np.array(X)
# Y_=np.array(Y)

# np.save('X_train_data.npy', X_)
# np.save('Y_train_data.npy', Y_)




# ######################Load data from file ################################
# filepath_train='C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/melspectrogram_train_v2.csv'
# filepath_test='C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/melspectrogram_test_v2.csv'
# train_data_matrix = pd.read_csv(filepath_train,sep=',', engine='python')

# test_data_matrix = pd.read_csv(filepath_test,sep=',', engine='python')

X_ =np.load('X_train_data.npy')
Y_=np.load('Y_train_data.npy')

X_test_actual1 =np.load('X_test_data.npy')

X_train,X_test,Y_train,Y_test = train_test_split(X_, Y_, test_size=0.2, random_state=1)


X_train = np.transpose(X_train,[0,2,1])
X_test = np.transpose(X_test,[0,2,1])

print(X_train.shape)
print(X_test.shape)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

num_classes =Y_test.shape[1]

print(Y_train.shape)

print(Y_test.shape)

#Actual test samples#############################################################

# count = 0
# dirList1 = []
# # X_test_actual = []
# file_id =[]
# input_length =441000
# dirList1+=(glob.glob('C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 2/bird-audio-detection/audio/*.npy'))
# for path in dirList1:
#     count+=1
#     print(count)
#     file_id.append(int(path.split("/")[7].split("\\")[1].split('.')[0]))
# #     X_test_1=np.load(path)
    
#     X_test_1 = librosa.resample(X_test_1,orig_sr=48000,target_sr=44100)
    
#     if len(X_test_1) > input_length:
#             X_test_1 = X_test_1[:input_length]
#     else:
#             X_test_1 = np.pad(X_test_1, (0, max(0, input_length - len(X_test_1))), "constant")    
      
#     X_test_1 =X_test_1 *1/np.max(np.abs(X_test_1))
    
#     kwargs_for_mel = {'n_mels': 40,'fmax':22050}
#     x_actual = librosa.feature.melspectrogram(
#             y=X_test_1, 
#             sr=44100, 
#             n_fft=1024, 
#             hop_length=512, 
#             **kwargs_for_mel)
    
#     X_test_actual.append(np.array(x_actual))
    
#X_test_actual1 = np.array(X_test_actual)  

# np.save('X_test_data.npy', X_test_actual1)

# print(X_test_actual1.shape)



X_test_actual1 = np.transpose(X_test_actual1,[0,2,1])
print(X_test_actual1.shape)



#################################################################################
  
# #Definition of LSTM Model
# model = Sequential()
# model.add(LSTM(32, return_sequences = False, input_shape = (862,40)))
# model.add(Dense(2, activation = 'softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs = 25, batch_size = 32)

# y_predicted_lstm = model.predict_classes(X_test_actual1)

# accu_score = model.evaluate(X_test, Y_test, verbose=0)[1]*100
# print(accu_score)

# for i in range(0,4512) :
#     output=[]
#     output.append(file_id[i])
#     output.append(y_predicted_lstm[i])
    
#     with open('predictions_lstm.csv', 'a', newline = '') as f1 :
#       writer = csv.writer(f1)
#       writer.writerow(output)
# f1.close()



####################CNN#######################################################
X_train = X_train[...,np.newaxis]
X_test = X_test[...,np.newaxis]
X_train = X_train.reshape((X_train.shape[0],862,40,1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0],862,40,1)).astype('float32')
X_test_actual1 = X_test_actual1[...,np.newaxis]
X_test_actual1 = X_test_actual1.reshape((X_test_actual1.shape[0],862,40,1)).astype('float32')




all_layers = [
          TimeDistributed(Conv2D(kernel_size = (3,3), filters = 96, activation='relu', input_shape=(862, 40,1), padding='same')),
          TimeDistributed(BatchNormalization()),
          TimeDistributed(MaxPooling2D(pool_size=(1, 5))),
          Dropout(0.25),
         TimeDistributed( Conv2D(kernel_size = (3,3), filters = 96, activation='relu',  padding='same')),
         TimeDistributed( BatchNormalization()),
         TimeDistributed( MaxPooling2D(pool_size=(1, 2))),  
          Dropout(0.25),
        TimeDistributed(  Conv2D(kernel_size = (3,3), filters = 96, activation='relu',  padding='same')),
        TimeDistributed(  BatchNormalization()),
        TimeDistributed(  MaxPooling2D(pool_size=(1, 2))), 
          Dropout(0.25),
        TimeDistributed(  Conv2D(kernel_size = (3,3), filters = 96, activation='relu',  padding='same')), 
        TimeDistributed(  BatchNormalization()),
        TimeDistributed(  MaxPooling2D(pool_size=(1, 2))),
          Dropout(0.25),
         #Permute((2,1)),    
         # Reshape((862,96), input_shape=(862,1,96),name='reshape'),
         LSTM(96, return_sequences=False),
         LSTM(96, return_sequences=False), 
         # MaxPooling1D(),    
          Flatten(), 
          Dense(100, activation = 'relu'),               
          Dense(2, activation = 'softmax')]

# Compile model
model = Sequential(all_layers)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train,validation_data = (X_test, Y_test),batch_size=32, epochs = 25)

y_predicted_cnn = model.predict_classes(X_test_actual1)


for i in range(0,4512) :
    output=[]
    #output.append(file_id[i])
    output.append(y_predicted_cnn[i])
    
    with open('predictions_crnn_1.csv', 'a', newline = '') as f1 :
      writer = csv.writer(f1)
      writer.writerow(output)
f1.close()


###Creating prediction csv file #################################################























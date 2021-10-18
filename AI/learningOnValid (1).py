# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:31:34 2017

@author: Nick Adams
"""

#This part of the learning is to give it a head start on some good moves played by proffesionals. I had a data set of around 450k moves made by proffesional players. It gets given a board setup and then it will return what it thinks is the best moves to make in that situation. The second part of the data is all the moves that the pros made in that board stepup, so it then compares what the network says is best with what the pros did and it learns off whether it makes the same moves as the pros. It learns off whether the output is the same as what it should be outputting.

import pickle
#import tensorflow as tf
import numpy as np
import h5py


from keras.regularizers import l2 , activity_l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
#from keras import backend as K

print("started")

#file = open("validMoves.txt","rb") # use small moves for testing
#moves = pickle.load(file)
#file.close()
#
#
#file = open("validPos.txt","rb") # use small moves for testing
#pos = pickle.load(file)
#file.close()

#loads the game data 
moves = np.load('bds_out.npy')
pos = np.load('pos_out.npy')


boards = np.empty((len(moves),8,8))

#converts the board into an array 
for i in range(len(moves)):
    boards[i][:][:] = np.asarray(moves[i][:][:]).reshape(8,8)/2
          
#for i in range(5):
#    bd = boards[i][:][:][:]
#    b = '\n'.join('\t'.join('%0.1f' %x for x in y) for y in bd)
#    print(b)
#    print('\n')


movesLen = int(len(moves))
print(len(moves))
print(int(len(moves)))

  
csv_logger = CSVLogger('training.csv')

#saves the weights after every epoch
checkPoint = ModelCheckpoint('weights_1024.h5',verbose=1 ,save_best_only=False, save_weights_only=True ,period=1)
tBoard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)



batch_size = 128 # 128
nb_epoch = 10000

wt_init = 'glorot_normal'
#Convolutional grid size
kernel_size = (2, 2)
#Convolutional filters
nb_filters = 32
#size of input board array size
input_shape = (8, 8, 1)
#size of network layers #### This is what i need to change and adjust
denseSize = 512
#size of moves output array
movesOutput = 32




movesLen = int(len(moves))

learningLen = int(movesLen)

ranMoves = np.random.choice(movesLen,size=movesLen,replace=False)

#creates a seperate batch for learning and testing on, this allows for generalisation as it doesn't just learn on the board postions it has seen
learningList , testingList = np.split(ranMoves,[learningLen])

X_train = boards[learningList][:][:]
Y_train = pos[learningList][:][:]

#converts into shape that tensorflow will accept
X_train = X_train.reshape(X_train.shape[0],8,8,1)
print(X_train.shape,Y_train.shape)

#converts to float32 so it can run quickly off a graphics card
X_train = X_train.astype('float32')
Y_train = Y_train.astype('float32')


X_test = boards[testingList][:][:] ##change to testingList for real learning
Y_test = pos[testingList][:][:] ## 

#converts into shape that tensorflow will accept
X_test = X_test.reshape(X_test.shape[0],8,8,1)
print(X_test.shape,Y_test.shape)

#converts to float32 so it can run quickly off a graphics card
X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')


#This is how the network is set up, we define what we want as the different layers and how big they are are what type of activation they use. This is what needs to be changed if the network doesn't learn. It may be set too small so it doesn't have enough room to learn.
model = Sequential()
#the convolution layer samples the input in a 2x2 grid and tries to find patterns in the data, this is good for checkers as 2x2 grids can be made in different places on the board and share the same properties. This means if it sees a certain 2x2 configuration it should be apply to apply the same logic to the same 2x2 grid somewhere else on the board 
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],subsample=(2,2),border_mode='valid',input_shape=input_shape,init = wt_init))
model.add(Activation('relu'))
#This flattens the network into a one dimensional layer
model.add(Flatten())
#layers are then added with the denseSize length and relu activation. The number or layers and size of each layer can be changed to change how the network is set out. There's no way to know exactly the best setup for the network so it's trail and error
model.add(Dense(denseSize,init = wt_init)) # ,W_regularizer=l2(0.01)
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(denseSize,init = wt_init))
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(denseSize,init = wt_init))
model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
#The model then gets reshapes into 32x4 which is the size of the output
model.add(Reshape((32,4)))


#model = Model(input=main_input, output=main_output)

#model is then compiled
model.compile(loss = 'mse',
              optimizer='Adam',
              metrics=['accuracy']) # loss='categorical_crossentropy'
model.summary()

#saves the model so when it comes to reinforcement learning, it doesn't have to be defined again and can just be loaded in with the correct configuration
model.save("my_model_1024.h5")

#this will runs some learning and then test it on the test data and give data on the loss and accuracy of it. 
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1,shuffle=True, validation_data=(X_test, Y_test),callbacks=[csv_logger,tBoard, checkPoint])# , tBoard



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
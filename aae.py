# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:01:03 2017

@author: Star
"""

#import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
#mpl.use('Agg')

import pandas as pd
#import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np


def model_encoder():
    with tf.device('/gpu:1'):
        model = Sequential()
        #nch = 4096
        h = 5
        reg = lambda: l1l2(l1=1e-7,l2=1e-7)
        model.add(Convolution2D(128,h,h,border_mode='same',W_regularizer = reg(),input_shape=(640,640,3)))
        model.add(BatchNormalization(mode=0))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(32,h,h,border_mode='same',W_regularizer = reg()))
        model.add(BatchNormalization(mode=0))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Convolution2D(32,h,h,border_mode='same',W_regularizer = reg()))
        model.add(BatchNormalization(mode=0))
        model.add(LeakyReLU(0.2))
        model.add(MaxPooling2D(pool_size=(2,2)))
        return model
    
def model_decoder():
    with tf.device('/gpu:1'):
        model = Sequential()
        h = 5
        reg = lambda: l1l2(l1=1e-7,l2=1e-7)
        model.add(Convolution2D(32,h,h,border_mode='same',W_regularizer = reg(),input_shape=(80,80,32)))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Convolution2D(32,h,h,border_mode='same',W_regularizer = reg()))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Convolution2D(128,h,h,border_mode='same',W_regularizer = reg()))
        model.add(UpSampling2D(size=(2,2)))
        model.add(Convolution2D(3,h,h,border_mode='same',W_regularizer = reg()))
        return model

    
def model_discriminator():
    with tf.device('/gpu:1'):
        model = Sequential()
        nch = 128
        h = 5
        reg = lambda: l1l2(l1=1e-7, l2=1e-7)
        #c1 = Convolution2D(int(nch / 64), h, h, border_mode='same', W_regularizer=reg(),
        #                   input_shape=( 80, 80,3))
        #c2 = Convolution2D(int(nch / 32), h, h, border_mode='same', W_regularizer=reg())
        
        #c3 = Convolution2D(int(nch / 16), h, h, border_mode='same', W_regularizer=reg())
        
        c4 = Convolution2D(int(nch / 8), h, h, border_mode='same', W_regularizer=reg(),
                input_shape=(80,80,32))
        
        c5 = Convolution2D(int(nch / 4),h,h,border_mode='same',W_regularizer = reg())
        
        c6 = Convolution2D(int(nch / 2),h,h,border_mode='same',W_regularizer = reg())
        
        c7 = Convolution2D(nch,h,h,border_mode='same',W_regularizer=reg())
        
        c8 = Convolution2D(1,h,h,border_mode='same',W_regularizer = reg())

        model = Sequential()
        model.add(c4)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c5)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c6)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c7)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c8)
        model.add(AveragePooling2D(pool_size=(5, 5), border_mode='valid'))
        model.add(Flatten())
        model.add(Activation('sigmoid'))
        return model
    
def main():
    with tf.device('/gpu:1'):
        f = open(r'meitu_img.pkl','rb')
        train_image = pickle.load(f)
        f.close()
        f = open(r'valid_img.pkl','rb')
        train_label = pickle.load(f)
        f.close()      
        encoder = model_encoder()
        decoder = model_decoder()
        dis = model_discriminator()
        encoder.summary()
        decoder.summary()
        dis.summary()
        encoder_dis = Sequential()
        encoder_dis.add(encoder)
        dis.trainable = False
        encoder_dis.add(dis)
        encoder_dis.compile(loss='binary_crossentropy',optimizer='adam')
        model = Sequential()
        dis.trainable = True
        dis.compile(loss='binary_crossentropy',optimizer='adam')
        model.add(encoder)
        model.add(decoder)
        model.compile(loss='mae',optimizer='adam')
        
   
        for i in range(50):
            print('all epoch:'+str(i))
            
            for j in range(int(len(train_image)/4)):
                print('num pic:'+str(j*4)+'::'+str(i))
                train_image_batch = train_image[j*4:(j+1)*4]
                train_image_batch = np.asarray([img/255 for img in train_image_batch])
                train_label_batch = train_label[j*4:(j+1)*4]
                train_label_batch = np.asarray([img/255 for img in train_label_batch])
                z_fake = encoder.predict(train_image_batch)
                z_real = np.random.random((len(train_image_batch),80,80,32))
                y = [0]*len(train_image_batch)+[1]*len(train_image_batch)
                x = list(z_fake)+list(z_real)
                #encoder_dis.fit(x = x,y = y,epochs=5,batch_size=16)
                dis.fit(x = np.asarray(x),y = np.asarray(y),epochs=1,batch_size=12)
                encoder_dis.fit(x=train_image_batch,y = [1]*len(train_image_batch),epochs=1,batch_size=4)
                model.fit(x = train_image_batch,y = train_label_batch,epochs=1,
                        batch_size=4)
        
        encoder_dis.save('encoder_dis.hf5')
        dis.save('dis.hf5')
        model.save('ae2.hf5')
if __name__ == "__main__":
   with tf.device('/gpu:1'):
    	main()


import warnings

import cv2
from sklearn.utils import validation
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from keras.layers import BatchNormalization, Activation, Conv2D 
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.optimizers import Adam, RMSprop

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.read_images import read_images


# CONSTANTS
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=True)
input_shape = (299, 299)
num_channels = 3

def load_data(img_shape=(299, 299)):
    ids_and_genres = pd.read_csv("data/posters-and-genres.csv")
    ids_and_genres.drop(['Genre'], axis=1, inplace=True) # genre is irrelevant with boolean encoded genres
    
    

    ids = ids_and_genres['Id'].values
    genres = ids_and_genres.loc[:, ids_and_genres.columns != 'Id'].values

    # info = dict()
    # for i, id in enumerate(ids):
    #     info[id] = genres[i]

    imgs = read_images(ids)
    
    x_train, x_test, y_train, y_test = train_test_split(imgs, genres, test_size=0.2)
    
    return x_train, x_test, y_train, y_test

def train(model):
    x_train, y_train, x_test, y_test = load_data()

    checkpoint1 = ModelCheckpoint('xception_checkpoint-1.h5', 
                              save_freq='epoch', 
                              verbose=1, 
                              save_weights_only=True)
    checkpoint2 = ModelCheckpoint('xception_checkpoint-2.h5', 
                                save_freq='epoch', 
                                verbose=1, 
                                save_weights_only=True)
    checkpoint3 = ModelCheckpoint('xception_checkpoint-3-best.h5', 
                                save_freq='epoch', 
                                verbose=1, 
                                monitor='loss', 
                                save_best_only=True, 
                                save_weights_only=True)

    result = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), \
        callbacks=[checkpoint1, checkpoint2, checkpoint3])
    
    # Plot history
    plot_history(result)

def get_image_gen(info_arg,
                  batch_size=48,
                  shuffle=True, 
                  image_aug=True, 
                  eq_dist=False, 
                  n_ref_imgs=16, 
                  crop_prob=0.5, 
                  crop_p=0.5):
    if image_aug:
        datagen = ImageDataGenerator(
            rotation_range=4.,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            channel_shift_range=25,
            horizontal_flip=True,
            fill_mode='nearest')
        
        if crop_prob > 0:
            datagen_crop = ImageDataGenerator(
                rotation_range=4.,
                shear_range=0.2,
                zoom_range=0.1,
                channel_shift_range=20,
                horizontal_flip=True,
                fill_mode='nearest')
        
    count = len(info_arg)
    while True:
        if eq_dist:
            def sample(df):
                return df.sample(min(n_ref_imgs, len(df)))
            info = info_arg.groupby('landmark_id', group_keys=False).apply(sample)
        else:
            info = info_arg
        print('Generate', len(info), 'for the next round.')
        
        #shuffle data
        if shuffle and count >= len(info):
            info = info.sample(frac=1)
            count = 0
            
        # load images
        for ind in range(0,len(info), batch_size):
            count += batch_size

            y = info['landmark_id'].values[ind:(ind+batch_size)]
            
            imgs = read_images(info.iloc[ind:(ind+batch_size)])
            if image_aug:
                cflow = datagen.flow(imgs, 
                                    y, 
                                    batch_size=imgs.shape[0], 
                                    shuffle=False)
                imgs, y = next(cflow)             

            imgs = preprocess_input(imgs)
    
            y_l = label_encoder.transform(y[y>=0.])        
            y_oh = np.zeros((len(y), n_cat))
            y_oh[y >= 0., :] = one_hot_encoder.transform(y_l.reshape(-1,1)).todense()
                    
            yield imgs, y_oh
 

def generalized_mean_pool_2d(X, gm_exp=tf.Variable(3., dtype=tf.float32)):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                           axis=[1,2], 
                           keepdims=False)+1.e-8)**(1./gm_exp)
    return pool

def batch_GAP(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)    
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)
    
    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)

    GAP = tf.reduce_mean(
          tf.cumsum(is_c) * is_c / tf.cast(
              tf.range(1, n_pred + 1), 
              dtype=tf.float32))
    
    return GAP    

def binary_crossentropy_n_cat(y_t, y_p):
    return keras.metrics.binary_crossentropy(y_t, y_p) * n_cat

def main():
    # K.clear_sessions()
    n_cat = 81313

    model = Xception(input_shape=list(input_shape) + [num_channels], weights="imagenet", include_top=False)
    # model.summary()
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:85]:
        layer.trainable = False
    
    # model.summary()
    gm_exp = tf.Variable(3., dtype=tf.float32)
    x_feat = Input(model.output_shape[1:])

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    X = lambda_layer(x_feat)
    X = Dropout(0.05)(X)
    X = Activation('relu')(X)
    X = Dense(n_cat, activation='softmax')(X)

    top_model = Model(inputs=x_feat, outputs=X)
    # top_model.summary()
    x_image = Input(list(input_shape) + [3])

    x_f = model(x_image)
    x_f = top_model(x_f)

    model = Model(inputs=x_image, outputs=x_f)
    # model.summary()

    optimizer = Adam(lr=0.0001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=[binary_crossentropy_n_cat, 'accuracy', batch_GAP])
    train(model)

if __name__ == "__main__":
    main()
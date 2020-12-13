import os
import numpy as np
import pandas as pd
from utils.read_images import read_images
from utils.misc import get_genres
from sklearn.model_selection import train_test_split

def split_data(data="data/posters-and-genres.csv", img_shape=(299, 299)):
    '''
    read data from the csv file containing the id of movies and the labels

    Keyword Arguments
    ==========
    data:
        location of poster dataset

    img_shape:
        (299, 299) for input size to XceptionNet

    Return
    ==========
    tuple of numpy arrays: (x_train, x_test, y_train, y_test)
    '''
    imgs, genres = load_data(data, img_shape=img_shape)

    # call the sklearn train_test_split method
    x_train, x_test, y_train, y_test = train_test_split(imgs, genres, test_size=0.2, random_state=3520)
    
    return x_train, x_test, y_train, y_test

def load_train_test(training_data='data/train_data.csv', testing_data='data/test_data.csv', img_shape=(299,299)):
    '''
    top level method for getting the training and test data for the CNN models

    Parameters
    ==========
    `training_data`:
        path to csv file containing training data

    `testing_data`:
        path to csv file containing training data

    Return
    ==========
    x_train, y_train, x_test, y_test
    '''
    x_train, y_train, _, genres = load_data(training_data, img_shape=img_shape) 
    x_test, y_test, _, _ = load_data(testing_data, img_shape=img_shape)
    return x_train, y_train, x_test, y_test, genres

def load_data(data='data/test_data.csv', posters_csv="data/posters-and-genres.csv", img_shape=(299,299)):
    '''
    load and read data for testing or training or other tasks associated with CNNs, where the 
    data is specified by the `data` parameter. Method merges `data` with `posters_csv` and 
    extracts the matching ImdbIDs, then reads the images and saves them into a numpy array.

    Parameters
    ==========
    `data`:
        any csv containing column imdbID, and whose imdbIDs are in `posters_csv`

    `posters_csv`:
        csv with data on poster image id and encoded genres

    Return
    ==========
    (imgs, labels, img_ids, genres)

    numpy array of images, numpy array of labels per image, numpy array of image ids, and a list
    of the column names of the labels 
    '''
    data_ids = pd.read_csv(data)['imdbID'].values
    ids_and_genres = pd.read_csv(posters_csv).drop(columns=['Genre'])
    ids_and_genres = ids_and_genres.loc[ids_and_genres['Id'].isin(data_ids)]
    img_ids = ids_and_genres['Id'].values # isolate the ids
    labels = ids_and_genres.loc[:, ids_and_genres.columns != 'Id'].values # isolate the genre labels

    print("\nLoading images........")
    # read in all the images into an array, return number of indices used (used to combat memory error)
    imgs, subset_size = read_images(img_ids, dimensions=img_shape)
    print("DONE!\n")

    # if there was a memory error, update the labels as were updated within read_images functions
    labels = labels[:subset_size]
    # genres = ids_and_genres.columns[-labels.shape[1]:]
    genres = get_genres()
    return imgs, labels, img_ids, genres
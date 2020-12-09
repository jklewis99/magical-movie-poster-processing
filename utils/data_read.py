import os
import numpy as np
import pandas as pd
from utils.read_images import read_images
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
    imgs, genres = load_data(data)

    # call the sklearn train_test_split method
    x_train, x_test, y_train, y_test = train_test_split(imgs, genres, test_size=0.2, random_state=3520)
    
    return x_train, x_test, y_train, y_test


def load_data(data='data/test_data.csv', posters_csv="data/posters-and-genres.csv"):
    '''
    read data for testing or training, as specified by the `data` parameter. Method merges
    `data` with `posters_csv` and extracts the matching ImdbIDs, then reads the images and saves
    them into a numpy array.

    Parameters
    ==========
    `data`:
        any csv containing column imdbID, and whose imdbIDs are in `posters_csv`

    `posters_csv`:
        csv with data on poster image id and encoded genres

    Return
    ==========
    numpy array of images, numpy array of labels per image, numpy array of image ids, and a list
    of the column names of the labels 
    '''
    data_ids = pd.read_csv(data)['imdbID'].values
    ids_and_genres = pd.read_csv(posters_csv).drop(columns=['Genre'])
    ids_and_genres = ids_and_genres.loc[ids_and_genres['Id'].isin(data_ids)]
    ids = ids_and_genres['Id'].values # isolate the ids
    genres = ids_and_genres.loc[:, ids_and_genres.columns != 'Id'].values # isolate the genre labels

    print("\nLoading images........")
    # read in all the images into an array, return number of indices used (used to combat memory error)
    imgs, subset_size = read_images(ids)
    print("DONE!\n")

    # if there was a memory error, update the labels as were updated within read_images functions
    genres = genres[:subset_size]
    return imgs, genres, ids, ids_and_genres.columns[-genres.shape[1]:]
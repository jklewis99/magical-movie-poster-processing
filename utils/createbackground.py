import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    posters = pd.read_csv("../data/posters-and-genres.csv").rename(columns={'Id': 'imdbID'})
    movie_metadata = pd.read_csv("../data/movies-metadata.csv", thousands=",")[['imdbID', 'imdbVotes', 'imdbRating', 'Title', 'Released']]
    print(movie_metadata.shape)
    movie_metadata = movie_metadata.merge(posters, on="imdbID")
    print(movie_metadata.shape)
    movie_metadata = movie_metadata.apply(create_year, axis=1)
    # movie_metadata.imdbVotes = movie_metadata.imdbVotes.astype(int)
    m = movie_metadata['imdbVotes'].quantile(0.9)
    c = movie_metadata['imdbRating'].mean()
    movie_metadata['score'] = movie_metadata.apply(lambda x: weighted_rating(x, m, c), axis=1)
    yearly_group = movie_metadata.sort_values(['year_released', 'score'], ascending=[
        False, False])
    idx = yearly_group.groupby(['year_released'], sort=True)['score'].transform(max) == yearly_group['score']
    print(movie_metadata.head())
    highest_rated_per_year = yearly_group[idx].sort_values(['year_released'])
    make_image_stack(highest_rated_per_year)

def make_image_stack(dataframe):
    img_dir = "../data/Images/"
    img_ids = dataframe['imdbID'].values

    img_stack = []
    num_rows = 4
    num_cols = len(img_ids)//4
    plt.figure()
    for i in range(num_rows):
        row = []
        for id in img_ids[num_cols*i:num_cols*(1+i)]:
            print(id)
            img = cv2.imread(os.path.join(img_dir, id + ".jpg"))
            print(img.shape)
            # plt.imshow(img)
            # plt.show()
            img = cv2.resize(img, (300, 450), interpolation=cv2.INTER_AREA)
            row.append(img)
        if len(row) == num_cols:
            img_stack.append(np.concatenate(row, axis=1))

    img_stack = np.concatenate(img_stack, axis=0)
    plt.figure()
    plt.imshow(img_stack)
    plt.show()
    cv2.imwrite("background-img.png", img_stack)


def create_year(series_object):
    # print(series_object)
    if isinstance(series_object['Released'], str):
        year = int(series_object['Released'].split()[-1])
    else:
        year = 0
    series_object['year_released'] = year
    return series_object

def weighted_rating(x, m, c):
    v = x['imdbVotes']
    R = x['imdbRating']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * c)

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    metadata = pd.read_csv("data/movies-metadata-cleaned.csv").drop(
        columns=['Language', 'Poster', 'Country', 'Director', 'Released', 'Writer', 'Genre', 'Actors'])
    ratings = pd.get_dummies(metadata['Rated'], prefix='rated') # one hot encode "Rated" column
    metadata = metadata.drop(columns=["Rated"]).join(ratings) # replace "Rated" with one_hot
    metadata = metadata.dropna() # drop the missing box_office values
    posters = pd.read_csv("data/posters-and-genres.csv").drop(columns=["Genre"]).rename(columns={"Id": "imdbID"})
    data = metadata.merge(posters, on='imdbID').drop_duplicates() # add genres
    data = data[((data['Short'] != 1) & ( data['N/A'] != 1))]
    data = data.drop(columns=['Reality-TV', 'Short', 'N/A'])
    cols = data.columns.tolist()
    cols = cols[1:2] + cols[5:6] + cols[2:5] + cols[6:] + cols[0:1]
    data = data[cols] # reorder columns
    # print(data.sum())
    train, test = train_test_split(data, test_size=0.2) # generate train and test data
    train.to_csv("data/train_data.csv", index=False)
    test.to_csv("data/test_data.csv", index=False)

if __name__ == "__main__":
    main()

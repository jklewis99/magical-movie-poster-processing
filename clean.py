import pandas as pd
import numpy as np

def main():
    '''
    all that really needs to be done is converting what were once string datatypes
    for values of Box_office and imdbVotes into floats (not integers because of NaN)
    '''
    df = pd.read_csv("movies-metadata.csv", thousands=",")
    df.astype({
        'Box_office': 'float32',
        'imdbVotes': 'float32'
    })
    df.to_csv("movies-metadata.csv")


if __name__ == "__main__":
    main()
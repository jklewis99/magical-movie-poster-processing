import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb


from scipy.stats import pearsonr

features = pd.read_csv("data/movies-metadata.csv", thousands=',')

features = features.dropna(axis=0)

revenue = features["Box_office"].values.astype(np.float)


imdb = features["imdbVotes"].values.astype(np.float)


correlation, _ = pearsonr(imdb, revenue)





##def load_feature(features, feature):
##
##    feature = features[feature].values
##
##    return feature
##
##
##
##def get_correlation(features, feature):
##    
##    correlation, _ = pearsonr(load_feature(features, feature), revenue) # correlation coefficient
##
##    return correlation



    
plt.figure()
plt.scatter(imdb, revenue)
plt.title("Revenue vs imdbVotes", fontsize=18)
plt.xlabel("imdbVotes", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation:.2f}", (np.min(imdb), 0.98*np.max(revenue)), fontsize=14) # plot the value on the graph
plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import os
import skimage.io as io

from scipy.stats import pearsonr

# -------------------------- movies-metadata.csv -------------------------------
# Read csv
features = pd.read_csv("data/movies-metadata.csv", thousands=',')

# Drop first row
features = features.dropna(axis=0)

# Extract values of columns needed
revenue = features["Box_office"].values.astype(np.float)
imdb = features["imdbVotes"].values.astype(np.float)
imdbRating = features["imdbRating"].values.astype(np.float)

# Evaluation Pearson r correlation
correlation, _ = pearsonr(imdb, revenue)

# Figure 1: Shows correlation between columns
plt.figure(1, figsize = (20, 7))
plt.subplot(2, 2, 1)
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
plt.scatter(imdb, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Votes", fontsize=16)
plt.xlabel("imdb Votes", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation:.2f}", (np.min(imdb), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation2, _ = pearsonr(imdbRating, revenue)

# Figure 2: Shows correlation between columns
plt.subplot(2, 2, 2)
plt.scatter(imdbRating, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Rating", fontsize=16)
plt.xlabel("imdb Rating", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation2:.2f}", (np.min(imdbRating), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# ------------------------------------------------------------------------------

# -------------------------- movies-metadata-cleaned.csv -----------------------

# Read csv
features = pd.read_csv("data/movies-metadata-cleaned.csv", thousands=',')

# Drop first row
features = features.dropna(axis=0)

# Extract values of columns needed
revenue = features["Box_office"].values.astype(np.float)
imdb = features["imdbVotes"].values.astype(np.float)
imdbRating = features["imdbRating"].values.astype(np.float)

# Evaluation Pearson r correlation
correlation, _ = pearsonr(imdb, revenue)

# Figure 3: Shows correlation between columns
plt.figure(2, figsize = (20, 7))
plt.subplot(2, 2, 1)
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
plt.scatter(imdb, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Votes", fontsize=16)
plt.xlabel("imdb Votes", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation:.2f}", (np.min(imdb), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation2, _ = pearsonr(imdbRating, revenue)

# Figure 4: Shows correlation between columns
plt.subplot(2, 2, 2)
plt.scatter(imdbRating, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Rating", fontsize=16)
plt.xlabel("imdb Rating", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation2:.2f}", (np.min(imdbRating), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# ------------------------------------------------------------------------------

plt.show()

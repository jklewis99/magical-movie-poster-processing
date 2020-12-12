import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import os
import skimage.io as io

from scipy.stats import pearsonr

DATAPATH = os.path.expanduser('~/Documents/PythonLibrary/csc3520/Project/magical-movie-poster-processing/data/')
# -------------------------- train_data.csv -------------------------------

# Read csv
path = os.path.join(DATAPATH, "train_data.csv")
features = pd.read_csv(path, thousands=',')

# Drop first row
features = features.dropna(axis=0)

# Extract values of columns needed
revenue = features["Box_office"].values.astype(np.float)
imdb = features["imdbVotes"].values.astype(np.float)
imdbRating = features["imdbRating"].values.astype(np.float)
runtime = features["Runtime"].values.astype(np.int)
metascore = features["Metascore"].values.astype(np.float)
releaseMonth = features["Release_month"].values.astype(np.int)

# Evaluation Pearson r correlation
correlation, _ = pearsonr(imdb, revenue)

# Figure 1: Shows correlation between columns
plt.figure(1, figsize = (20, 7))
plt.subplot(2, 3, 1)
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
plt.scatter(imdb, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Votes", fontsize=16)
plt.xlabel("imdb Votes", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation:.2f}", (np.min(imdb), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation2, _ = pearsonr(imdbRating, revenue)

# Figure 2: Shows correlation between columns
plt.subplot(2, 3, 2)
plt.scatter(imdbRating, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs imdb Rating", fontsize=16)
plt.xlabel("imdb Rating", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation2:.2f}", (np.min(imdbRating), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation3, _ = pearsonr(runtime, revenue)

# Figure 2: Shows correlation between columns
plt.subplot(2, 3, 3)
plt.scatter(runtime, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs Runtime", fontsize=16)
plt.xlabel("Runtime", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation3:.2f}", (np.min(runtime), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation4, _ = pearsonr(metascore, revenue)

# Figure 2: Shows correlation between columns
plt.subplot(2, 3, 4)
plt.scatter(metascore, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs Metascore", fontsize=16)
plt.xlabel("Metascore", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation4:.2f}", (np.min(metascore), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

# Evaluation Pearson r correlation
correlation5, _ = pearsonr(releaseMonth, revenue)

# Figure 2: Shows correlation between columns
plt.subplot(2, 3, 5)
plt.scatter(releaseMonth, revenue, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
plt.title("Revenue vs Release Month", fontsize=16)
plt.xlabel("Release Month", fontsize=14)
plt.ylabel("Revenue", fontsize=14)
plt.annotate(f"Pearson-R = {correlation5:.2f}", (np.min(releaseMonth), 0.98*np.max(revenue)), fontsize=12) # plot the value on the graph

plt.show()

# ------------------------------------------------------------------------------

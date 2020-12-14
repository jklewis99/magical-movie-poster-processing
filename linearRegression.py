
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

def linear(x_train, y_train):

    # Train the model with LinearRegression
    regressor = LinearRegression()
    return (regressor.fit(x_train, y_train))

    # Predict testing data
    preds = regressor.predict(x_test)

    # Comparison between training and testing metric
    # Coefficient of determination: 1 is perfect prediction
    r_squared = r2_score(y_test, preds)
    print(" Full: ", r_squared)

    # Plot results
    plt.figure()
    plt.scatter(preds, y_test, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")

    # Plot line from min point to max point
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # slope of line
    x1 = xlim[0]
    y1 = ylim[0]
    x2 = xlim[1]
    y2 = ylim[1]
    m = (y2 - y1) / (x2 - x1)

    regression_line = []
    for x in preds:
        regression_line.append((m*x))

    plt.plot(preds, regression_line, color = 'red', linewidth=1)
    plt.title('Actual Revenue vs Predicted Revenue', fontsize = 16)
    plt.ylabel('Actual Revenue', fontsize = 14)
    plt.xlabel('Predicted Revenue', fontsize = 14)
    plt.annotate(f"r2 Score = {r_squared:.2f}", (np.min(preds), 0.98*np.max(y_test)), fontsize=10) # plot the value on the graph
    plt.show()

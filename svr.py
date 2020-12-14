from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

def support_vector_regression(kernel, x_train, y_train):

    scalar_x = StandardScaler() # need to scale our data (I think)
    scalar_y = StandardScaler() # need to scale our data (I think)

    x_train = scalar_x.fit_transform(x_train)
    y_train = scalar_y.fit_transform(y_train.reshape(-1, 1))
    # x_test = scalar_x.transform(x_test)
    # y_test = scalar_y.transform(y_test.reshape(-1, 1))

    # Train the model with SVR
    regressor = SVR(kernel)

    # return regressor, scalar_x, and scalar_y
    return (regressor.fit(x_train, y_train.flatten())), scalar_x, scalar_y

    preds = regressor.predict(x_test)
    r_squared = r2_score(y_test, preds)

    rescaled_preds, rescaled_actual = inverse_transform(preds, y_test, scalar_y)
    plt.figure()
    plt.scatter(rescaled_preds, rescaled_actual, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")

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
    for x in rescaled_preds:
        regression_line.append((m*x))

    plt.plot(rescaled_preds, regression_line, color = 'red', linewidth=1)
    plt.title('Actual Revenue vs Predicted Revenue (' + kernel + ")", fontsize = 14)
    plt.ylabel('Actual Revenue', fontsize = 12)
    plt.xlabel('Predicted Revenue', fontsize = 12)
    plt.annotate(f"r2 Score = {r_squared:.5f}", (np.min(rescaled_preds), 0.98*np.max(rescaled_actual)), fontsize=10) # plot the value on the graph
    # Plot results
    plt.show()

def inverse_transform(predictions, actual, scalar):
    '''
    transform the values to original scale
    Return
    ==========
    tuple of rescaled `predictions`, rescaled `actual`
    '''
    if len(predictions.shape) > 1:
        rescaled_predictions = scalar.inverse_transform(np.array([val[0] for val in predictions]))
    else:
        rescaled_predictions = scalar.inverse_transform(predictions)
    rescaled_actual = scalar.inverse_transform(actual).flatten()
    return rescaled_predictions, rescaled_actual

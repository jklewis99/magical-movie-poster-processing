import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import pickle
import randomForests
import svr
import linearRegression
import argparse
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description=" Predict movie box office revenue.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    'model',
    choices=['linear', 'svr', 'rfr'],
    help="This is a REQUIRED PARAMETER to set mode to 'linear' or 'svr', 'rfr'")

parser.add_argument(
    '--kernel', '-kernel',
    choices=['linear', 'poly', 'rbf', 'sigmoid'],
    default='linear',
    type = str,
    help="If model is 'svr', this is a PARAMETER to specify which kernel to use. Default is 'linear':\
        \n'linear',\
        \n'poly',\
        \n'rbf',\
        \n'sigmoid'")

# This function loads data
def load_data():

    # Read csv
    training_Data = pd.read_csv("data/train_data.csv", thousands=',')
    testing_Data = pd.read_csv("data/test_data.csv", thousands=',')

    # Assign x and y training and testing values
    x_train = training_Data.iloc[ :, 2:-1].values.astype(np.float)
    y_train = training_Data[['Box_office']].values.astype(np.float)
    x_test = testing_Data.iloc[ :, 2:-1].values.astype(np.float)
    y_test = testing_Data[['Box_office']].values.astype(np.float)

    return x_train, y_train, x_test, y_test

def plot_results(r_squared, preds, y_test):

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

    plt.savefig('model.png')
    plt.show()

def predict(regressor, x_test):
    preds = regressor.predict(x_test)
    return preds

def main(args):
    '''
    Main function to define what functions to run
    '''
    model = args.model
    kernel = args.kernel

    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Check if we have to scale data
    scalar_x, scalar_y = None, None

    # Check which model, get regressor
    # If model == linear
    if model == 'linear':
        regressor = linearRegression.linear(x_train, y_train)

    # if model == randomForests
    elif model == 'rfr':
        regressor = randomForests.rfr(x_train, y_train, 30)

    # if model is Support Vector regression
    else:
        regressor, scalar_x, scalar_y = svr.support_vector_regression(kernel, x_train, y_train)

    # if scalar_x is presented
    if scalar_x:
        #scale data
        x_test = scalar_x.transform(x_test)
        pass

    preds = predict(regressor, x_test) # predict

    if scalar_y:
        preds, _ = svr.inverse_transform(preds, y_test, scalar_y)

    r2 = r2_score(y_test, preds) #get r2 score

    plot_results(r2, preds, y_test)

if __name__ == '__main__':
    try:
        main(parser.parse_args())
    except Exception as e:
        print(e)

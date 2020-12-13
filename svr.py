from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

def main():

    # Read csv
    trainingData = pd.read_csv("data/train_data.csv", thousands=',')
    testingData = pd.read_csv("data/test_data.csv", thousands=',')

    # Drop the first row
    trainingData = trainingData.dropna(axis=0)
    testingData = testingData.dropna(axis=0)

    # Assign x and y training and testing values
    x_train = trainingData.iloc[ :, 2:43].values.astype(np.float)
    y_train = trainingData[['Box_office']].values.astype(np.float)
    x_test = testingData.iloc[ :, 2:43].values.astype(np.float)
    y_test = testingData[['Box_office']].values.astype(np.float)

    results_r2 = []

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    # Train the model with SVR
    for kernel in kernels:
        regressor = SVR(kernel=kernel)
        regressor.fit(x_train, y_train.flatten())
        preds = regressor.predict(x_test)
        r_squared = r2_score(y_test, preds)
        results_r2.append(r_squared)

        plt.figure()
        plt.scatter(preds, y_test, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
        plt.title('Actual Revenue vs Predicted Revenue (' + kernel + ")", fontsize = 14)
        plt.ylabel('Actual Revenue', fontsize = 12)
        plt.xlabel('Predicted Revenue', fontsize = 12)
        plt.annotate(f"r2 Score = {r_squared:.5f}", (np.min(preds), 0.98*np.max(y_test)), fontsize=10) # plot the value on the graph

    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)),axis=1)
    print(table_of_results)

    # Plot results
    plt.show()


if __name__ == "__main__":
    main()

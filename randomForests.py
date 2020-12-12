from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

def main():
    # Read csv
    trainingData = pd.read_csv("data/train_data.csv", thousands=',')
    testingData = pd.read_csv("data/train_data.csv", thousands=',')

    # Drop the first column
    trainingData = trainingData.dropna(axis=0)
    testingData = trainingData.dropna(axis=0)

    # Assign x and y training and testing values
    xtrain = trainingData.iloc[ :, 2:43].values.astype(np.float)
    ytrain = trainingData[['Box_office']].values.astype(np.float)
    xtest = testingData.iloc[ :, 2:43].values.astype(np.float)
    ytest = testingData[['Box_office']].values.astype(np.float)

    # Ravel y_train
    ytrain = ytrain.ravel()

    # Number of trees of comparison
    numTrees = np.arange(20, 100, 5)
    results_r2 = []
    # Initialize "bests"
    best_preds = None
    best_r_squared = 0
    best_tree_count = 0

    for numTree in numTrees:
        # Train the model
        regressor = RandomForestRegressor(n_estimators= numTree, random_state = 18, max_samples = 0.8)

        # Fit the regressor to the training data
        regressor.fit(xtrain, ytrain)

        # Predict testing data
        preds = regressor.predict(xtest)

        # Comparison between training and testing metric
        # Coefficient of determination: 1 is perfect prediction
        r_squared = r2_score(ytest, preds)

        # update "bests", if this forest outperformed the current best
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_preds = preds
            best_tree_count = numTree
        results_r2.append(r_squared)

    # results
    results_r2 = np.array(results_r2)
    tableOfResults = np.concatenate((numTrees.reshape(len(numTrees), 1), results_r2.reshape(len(results_r2), 1)), axis=1)

    print(tableOfResults)
    #print(preds)

    # Plot results
    plt.figure()
    plt.scatter(preds, ytest, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
    plt.title('Actual Revenue vs Predicted Revenue', fontsize = 16)
    plt.ylabel('Actual Revenue', fontsize = 14)
    plt.xlabel('Predicted Revenue', fontsize = 14)
    plt.annotate(f"r2 Score = {best_r_squared:.2f}", (np.min(preds), 0.98*np.max(ytest)), fontsize=10) # plot the value on the graph
    plt.show()

if __name__ == "__main__":
    main()

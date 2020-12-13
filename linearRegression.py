
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
    xtrain = trainingData.iloc[ :, 2:43].values.astype(np.float)
    ytrain = trainingData[['Box_office']].values.astype(np.float)
    xtest = testingData.iloc[ :, 2:43].values.astype(np.float)
    ytest = testingData[['Box_office']].values.astype(np.float)

    # Train the model with LinearRegression
    regressor = LinearRegression()
    regressor.fit(xtrain, ytrain)

    # Predict testing data
    preds = regressor.predict(xtest)

    # Comparison between training and testing metric
    # Coefficient of determination: 1 is perfect prediction
    r_squared = r2_score(ytest, preds)
    print(" Full: ", r_squared)

    # Plot results
    plt.figure()
    plt.scatter(preds, ytest, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
    #plt.plot(xtest[:,0], preds, color = 'blue', linewidth=1)
    plt.title('Actual Revenue vs Predicted Revenue', fontsize = 16)
    plt.ylabel('Actual Revenue', fontsize = 14)
    plt.xlabel('Predicted Revenue', fontsize = 14)
    plt.annotate(f"r2 Score = {r_squared:.2f}", (np.min(preds), 0.98*np.max(ytest)), fontsize=10) # plot the value on the graph
    plt.show()

if __name__ == "__main__":
    main()

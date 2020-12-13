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
    testingData = pd.read_csv("data/train_data.csv", thousands=',')

    # Drop the first row
    trainingData = trainingData.dropna(axis=0)
    testingData = trainingData.dropna(axis=0)

    # Assign x and y training and testing values
##    x_train = trainingData[['imdbVotes', 'Runtime', 'imdbRating', 'Metascore', 'Release_month', 'Oscar_noms', 'Oscar_wins', 'Golden_globe_noms', 'Golden_globe_wins',
##                            'BAFTA_noms', 'BAFTA_wins', 'Other_noms', 'Other_wins', 'Release_month', 'rated_G', 'rated_NOT RATED', 'rated_PG', 'rated_PG-13', 'rated_R',
##                            'rated_UNRATED', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
##                            'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']].values.astype(np.float)
    x_train = trainingData.iloc[ :, 2:43].values.astype(np.float)
    y_train = trainingData[['Box_office']].values.astype(np.float)
##    x_test = testingData[['imdbVotes', 'Runtime', 'imdbRating', 'Metascore', 'Release_month', 'Oscar_noms', 'Oscar_wins', 'Golden_globe_noms', 'Golden_globe_wins',
##                            'BAFTA_noms', 'BAFTA_wins', 'Other_noms', 'Other_wins', 'Release_month', 'rated_G', 'rated_NOT RATED', 'rated_PG', 'rated_PG-13', 'rated_R',
##                            'rated_UNRATED', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
##                            'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']].values.astype(np.float)
    x_test = trainingData.iloc[ :, 2:43].values.astype(np.float)
    y_test = testingData[['Box_office']].values.astype(np.float)

    results_r2 = []

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    # Train the model with SVR
    for kernel in kernels:
        preds, r_squared = test(kernel, x_train, x_test, y_train, y_test)
        results_r2.append(r_squared)
        plt.figure()
        plt.scatter(preds, y_test, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")
        plt.title('Actual Revenue vs Predicted Revenue (' + kernel + ")", fontsize = 16)
        plt.ylabel('Actual Revenue', fontsize = 14)
        plt.xlabel('Predicted Revenue', fontsize = 14)
        plt.annotate(f"r2 Score = {r_squared:.5f}", (np.min(preds), 0.98*np.max(y_test)), fontsize=10) # plot the value on the graph

    kernels = np.array(kernels)
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate(
        (kernels.reshape(len(kernels), 1), results_r2.reshape(len(results_r2), 1)),
        axis=1)
    print(table_of_results)

    # Plot results
    plt.show()

    

def test(kernel, x_train, x_test, y_train, y_test):
    '''
    fit the SVR model with specified `kernel` to training data and test the SVR on the test data
    Parameters
    ==========
    `kernel`:
        string that specifies the kernel to be used
    `x_train`:
        numpy array with all feature values for each sample in training data
    `x_test`:
        numpy array with all feature values for each sample in testing data
    `y_train`:
        numpy array with correct continuous value (revenue) for each training sample
    `y_test`:
        numpy array with correct continuous value (revenue) for each testing sample
    Return
    ==========
    predictions, r_squared
    '''
    regressor_temp = SVR(kernel=kernel)
    regressor_temp.fit(x_train, y_train.flatten())
    preds = regressor_temp.predict(x_test)
    r_squared = r2_score(y_test, preds)
    return preds, r_squared



if __name__ == "__main__":
    main()


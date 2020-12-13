from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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
<<<<<<< HEAD
##    x_test = testingData[['imdbVotes', 'Runtime', 'imdbRating', 'Metascore', 'Release_month', 'Oscar_noms', 'Oscar_wins', 'Golden_globe_noms', 'Golden_globe_wins',
##                            'BAFTA_noms', 'BAFTA_wins', 'Other_noms', 'Other_wins', 'Release_month', 'rated_G', 'rated_NOT RATED', 'rated_PG', 'rated_PG-13', 'rated_R',
##                            'rated_UNRATED', 'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
##                            'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']].values.astype(np.float)
=======
>>>>>>> 2f393f706bb161726dd60b77b25c2cd5c8619c59
    x_test = testingData.iloc[ :, 2:43].values.astype(np.float)
    y_test = testingData[['Box_office']].values.astype(np.float)

    scalar_x = StandardScaler() # need to scale our data (I think)
    scalar_y = StandardScaler() # need to scale our data (I think)

    x_train = scalar_x.fit_transform(x_train)
    y_train = scalar_y.fit_transform(y_train.reshape(-1, 1))
    x_test = scalar_x.transform(x_test)
    y_test = scalar_y.transform(y_test.reshape(-1, 1))


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

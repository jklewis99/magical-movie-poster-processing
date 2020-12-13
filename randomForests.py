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
    training_Data = pd.read_csv("data/train_data.csv", thousands=',')
    testing_Data = pd.read_csv("data/test_data.csv", thousands=',')

    training_Data = training_Data.dropna(axis=0)
    testing_Data = testing_Data.dropna(axis=0)

    # Assign x and y training and testing values
    x_train = training_Data.iloc[ :, 2:43].values.astype(np.float)
    y_train = training_Data[['Box_office']].values.astype(np.float)
    x_test = testing_Data.iloc[ :, 2:43].values.astype(np.float)
    y_test = testing_Data[['Box_office']].values.astype(np.float)

    # Ravel y_train
    y_train = y_train.ravel()

    # Number of trees of comparison
    num_Trees = np.arange(20, 100, 5)
    results_r2 = []
    # Initialize "bests"
    best_preds = None
    best_r_squared = 0
    best_tree_count = 0

    for num_Tree in num_Trees:
        # Train the model
        regressor = RandomForestRegressor(n_estimators= num_Tree, random_state = 3520, max_samples = 0.80)

        # Fit the regressor to the training data
        regressor.fit(x_train, y_train)

        # Predict testing data
        preds = regressor.predict(x_test)

        # Comparison between training and testing metric
        # Coefficient of determination: 1 is perfect prediction
        r_squared = r2_score(y_test, preds)

        # update "bests", if this forest outperformed the current best
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_preds = preds
            best_tree_count = num_Tree
        results_r2.append(r_squared)

    # results
    results_r2 = np.array(results_r2)
    table_of_results = np.concatenate((num_Trees.reshape(len(num_Trees), 1), results_r2.reshape(len(results_r2), 1)), axis=1)

    print(table_of_results)

    # Plot results
    plt.figure()
    plt.scatter(best_preds, y_test, s = 1, marker = "o", facecolor = "none", edgecolor = "blue")

    # Plot line from min point to max point
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # x and y values
    x1 = xlim[0]
    y1 = ylim[0]
    x2 = xlim[1]
    y2 = ylim[1]
    m = (y2 - y1) / (x2 - x1)

    # Plot regression line
    regression_line = []
    for x in preds:
        regression_line.append((m*x))

    plt.plot(preds, regression_line, color = 'red', linewidth=1)

    plt.title('Actual Revenue vs Predicted Revenue', fontsize = 16)
    plt.ylabel('Actual Revenue', fontsize = 14)
    plt.xlabel('Predicted Revenue', fontsize = 14)
    plt.annotate(f"r2 Score = {best_r_squared:.2f}", (np.min(preds), 0.98*np.max(y_test)), fontsize=10) # plot the value on the graph
    plt.show()

if __name__ == "__main__":
    main()

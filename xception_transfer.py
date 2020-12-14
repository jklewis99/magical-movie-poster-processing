import argparse
import os
import cv2

import tensorflow.keras as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from utils.read_images import read_images
from utils.data_read import load_data
from utils.misc import labels_to_text, show_labels_and_predictions, show_poster_and_genres, get_genres, LabelsPerfect

def split_data(data="data/posters-and-genres.csv", img_shape=(299, 299)):
    '''
    read data from the csv file containing the id of movies and the labels

    Keyword Arguments
    ==========
    data:
        location of poster dataset

    img_shape:
        (299, 299) for input size to XceptionNet

    Return
    ==========
    tuple of numpy arrays: (x_train, x_test, y_train, y_test)
    '''
    imgs, genres = load_data(data)

    # call the sklearn train_test_split method
    x_train, x_test, y_train, y_test = train_test_split(imgs, genres, test_size=0.2, random_state=3520)
    
    return x_train, x_test, y_train, y_test

def load_test_data(data='data/test_data.csv'):
    test_ids = pd.read_csv(data)['imdbID'].values
    ids_and_genres = pd.read_csv("data/posters-and-genres.csv").drop(columns=['Genre'])
    ids_and_genres = ids_and_genres.loc[ids_and_genres['Id'].isin(test_ids)][:10]
    ids = ids_and_genres['Id'].values[:10] # isolate the ids
    genres = ids_and_genres.loc[:, ids_and_genres.columns != 'Id'].values[:10] # isolate the genre labels

    print("\nLoading images........")
    # read in all the images into an array, return number of indices used (used to combat memory error)
    imgs, subset_size = read_images(ids)
    print("DONE!\n")

    # if there was a memory error, update the labels as were updated within read_images functions
    genres = genres[:subset_size]
    return imgs, genres, ids_and_genres

def generalized_mean_pool_2d(X, gm_exp=tf.Variable(3., dtype=tf.float32)):
    '''
    fancy method to make use of only salient regions in an image:
    take X to the number of channels (general mean exponent), `gm_emp`, take the 
    reduced mean, add an epsilon value, then take the `gm_emp`th root (cubed root)
    '''
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                           axis=[1,2], 
                           keepdims=False)+1.e-8)**(1./gm_exp)
    return pool  

def binary_crossentropy_multiclass(true_labels, predicted_labels, num_labels):
    '''
    calculate the binary cross entropy for a multi-labeled sample with one-hot-like encoded labels

    Parameters
    ==========
    true_labels
        the actual labels of the samples in the batch
    predicted_labels:
        the predicted labels of the samples in the batch
    num_classes:
        the number of classes/labels

    Return
    ==========
    the binary cross entropy calculated for each label
    '''
    return K.metrics.binary_crossentropy(true_labels, predicted_labels) * num_labels

def plot_history(result):
    # Create and save a plot consists of val_loss and val_acc
    plt.figure()
    plt.plot(result.history['val_loss'], label='val_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(os.getcwd(), 'Data', 'loss.png'))

    plt.figure()
    plt.plot(result.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epcohs")
    plt.ylabel("Accuracy")
    
    plt.savefig(os.path.join(os.getcwd(), 'Data', 'accuracy.png'))

def train(train_data="data/train_data.csv", test_data="data/test_data.csv", epochs=100, loss='binary_crossentropy'):
    '''
    train the model
    '''
    x_train, y_train, _, genres = load_data(train_data) 
    x_test, y_test, _, _ = load_data(test_data)
    num_labels = len(genres) # total number of genres
    model = build_model(num_labels)
    
    checkpoint1 = ModelCheckpoint('weights/xception_checkpoint-1.h5',
                              save_freq='epoch',
                              verbose=1,
                              save_weights_only=True)
    checkpoint2 = ModelCheckpoint('weights/xception_checkpoint-best.h5',
                                save_freq='epoch',
                                verbose=1, 
                                monitor='loss', 
                                save_best_only=True, 
                                save_weights_only=True)
    print("Checkpoints set.")
    
    print("Fit model on training data")
    # model.load_weights('weights/xception_checkpoint-best.h5')
    model.compile(optimizer='adam', loss=loss, metrics=[
                                                    lambda actual, preds: binary_crossentropy_multiclass(actual, preds, num_labels),
                                                    LabelsPerfect(num_labels),
                                                    'accuracy'])
    result = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint1, checkpoint2])
    # Plot history
    plot_history(result)

def test(weights_file, test_data="data/test_data.csv", path="data/Images", num_samples=3, save_imgs=False):
    '''
    test the model with a specified weights file
    '''
    imgs, actual_genres, ids, genre_names = load_data(test_data)
    samples = np.random.choice(len(imgs), num_samples, replace=False)
    num_genres = actual_genres.shape[1]
    model = load_model(weights_file, num_genres)
    predictions = model.predict(imgs[samples])
    text_preds, text_actuals = labels_to_text(predictions, actual_genres, genre_names)
    # variables for displaying
    for i, data_index in enumerate(samples):
        # get id, used to read picture
        img_id = ids[data_index]
        # use opencv to read the image
        img = cv2.imread(path + "/" + img_id + ".jpg")
        img = show_labels_and_predictions(img, text_preds[i], text_actuals[i])
        cv2.imshow("window", img)
        cv2.waitKey(0)
        if save_imgs:
            cv2.imwrite(f"figures/xception_preds/img_id.png")
    return predictions, actual_genres

def test_data_evaluation(model_path, model_type):
    '''
    This function finds the threshold for predicting genres on the test data

    Parameters
    ==========
    `model_path`:
        absolute or relative path to the model's file

    `model_type`:
        string type of the model
    '''
    model = load_model(model_path, 25)  # Load model
    x_test, y_test, _, genres = load_data()  # Load data
    thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    plot_data = []
    perf_data = []

    for threshold in thresholds:
        genres_accuracy, perfect_accuracy = evaluate(model, threshold, x_test, y_test, len(genres))
        plot_data.append(genres_accuracy)
        perf_data.append(perfect_accuracy)
        print(plot_data)

    fig = plt.figure()
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(thresholds, plot_data)
    graph.set_title(model_type + " Total Genre Accuracy Per Threshold")
    graph.set_xlabel('Threshold')
    graph.set_ylabel('Accuracy')
    plt.savefig(os.path.join('figures', model_type + '_evaluation.png'))
    #plt.show()
    fig = plt.figure()
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(thresholds, perf_data)
    graph.set_title(model_type + " Perfect Multi-label Classification Accuracy")
    graph.set_xlabel('Threshold')
    graph.set_ylabel('Accuracy')
    plt.savefig(os.path.join('figures', model_type + '_perfect_evaluation.png'))


def predict(img_path, model_path='weights/xception_checkpoint-best.h5'):
    '''
    This function predicts a specific poster st specified `img_path`

    Parameters
    ==========
    `img_path`:
        absolute or relative path to poster image

    `genres`:
        all possible string genres for a film (as defined in training)
    
    `model_path`:
        absolute or relative path to the model's file
    '''
    result = []  # This variable refers to all the genre that the poster might belong to
    genres = get_genres()
    model = load_model(model_path, len(genres))  # Load model
    data = np.zeros((1, 299, 299, 3))  # Initialize data
    data[0] = np.asarray(Image.open(img_path).resize((299, 299), Image.ANTIALIAS))  # Load data
    prediction = model.predict(data)  # Make prediction

    # Get the genre that is equal or larger than the threshold
    for i in range(len(genres)):
        if prediction[0][i] >= 0.2:
            result.append(genres[i])
    show_poster_and_genres(img_path, result)
    print('Genre(s): ' + str(result))

def load_model(weights_path, num_labels, input_shape=(299,299), num_channels=3):
    '''
    load a pretrained model
    '''
    model = build_model(num_labels)
    model.load_weights(weights_path)
    return model

def find_threshold():
    '''
    This function finds the threshold for predicting
    '''
    x_train, x_test, y_train, y_test = load_data()  # Load data
    model = load_model("xception_checkpoint-2-best.h5", y_train[1])  # Load model
    data = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    plot_data = []
    predictions = []

    for threshold in thresholds:
        evaluation = evaluate(model, threshold, data, label, 25)
        plot_data.append(evaluation[0])
        predictions.append(evaluation[1])

    fig = plt.figure()
    plt.plot(thresholds, plot_data, color='red')
    plt.title("Model Accuracy per Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Model Prediction Performance")
    plt.savefig(os.path.join('xceptionnet-evaluation.png'))
    plt.show()
    
    fig = plt.figure()
    plt.bar(np.arange(len(thresholds)), [x - 0.9 for x in plot_data], bottom=0.9, align='center', color='orange')
    plt.title("Model Accuracy per Threshold")
    plt.xticks(np.arange(len(thresholds)), thresholds)
    plt.xlabel("Threshold")
    plt.ylabel("Model Prediction Performance")
    plt.savefig(os.path.join('xceptionnet-evaluation-bar.png'))
    plt.show()
    
    return {
        "thresholds": thresholds,
        "plot_data": plot_data,
        "data": data,
        "labels": label,
        "predictions": predictions
    }

def evaluate(model, threshold, data, actual_labels, num):
    '''
    This function evaluates the model performance given the threshold and other data

    Parameters
    ==========
    `model`:
        Keras model for evaluating
        
    `threshold`:
        required confidence by model to count a prediction as a "yes" or "no"
    
    `data`:
        data from which to get predictions

    `actual_labels`:
        correct labels for the data

    `num_labels`:
        number of possible labels

    Return
    ==========
    percentage of labels accurately predicted by the model
    '''
    prediction = model.predict(data)
    acc = 0

    # TODO: EVALUATE THIS CODE BLOCK
    #######################################
    binary_preds = np.zeros_like(prediction)
    binary_preds[np.where(prediction > threshold)] = 1
    values = np.equal(binary_preds, actual_labels) # get the element-wise equality
    perfect_count = np.count_nonzero(np.all(values, axis=1)) # get which rows are equal for each genre
    perfect_accuracy = perfect_count / len(binary_preds)
    print(perfect_accuracy)
    #######################################
    
    for result, actual_label in zip(prediction, actual_labels):
        for i in range(num):
            if result[i] >= threshold and actual_label[i] == 1:
                acc += 1

            if result[i] < threshold and actual_label[i] == 0:
                acc += 1

    return acc / (len(actual_labels) * num), perfect_accuracy

# def evaluate(model, threshold, data, label, num_labels):
#     '''
#     This function evaluates the model performance given the threshold and other data
#     '''
#     print("Predicting.......", end="")
#     prediction = model.predict(data)
#     print("DONE")
#     acc = 0

#     for result, sub_label in zip(prediction, label):
#         for i in range(num_labels):
#             if result[i] >= threshold and sub_label[i] == 1:
#                 acc += 1

#             if result[i] < threshold and sub_label[i] == 0:
#                 acc += 1

#     return (acc / (len(label) * num_labels), prediction)

def build_model(num_labels, input_shape=(299, 299), num_channels=3):
    '''
    adapt the xception net model for prediction of this particular task
    This code was heavily adopted from a notebook on xceptionnet transfer learning
    nature image classification

    Parameters
    ==========    
    `num_labels`:
        labels for which to transfer the model's output
    
    Keyword Args
    ==========
    `input_shape`:
        (height, width) of image (required to be (299,299))
    
    `num_channels`:
        channels of image (required to be 3 for imagenet)

    Return
    ==========
    XceptionNet model with adjusted trainable weights and adjust MLP
    '''
    # use pretrained XceptionNet on imagenet dataset
    xception_model = Xception(input_shape=list(input_shape) + [num_channels], weights="imagenet", include_top=False)
    # model.summary()
    for layer in xception_model.layers:
        layer.trainable = True
    for layer in xception_model.layers[:85]:
        layer.trainable = False
    # model.summary()
    generalized_mean_exponent = tf.Variable(3., dtype=tf.float32)
    xception_output_feature_vector = Input(xception_model.output_shape[1:])

    # compute the generalize mean of the axes of the output layer in xceptionnet, to be used
    # as the first part of the fully connected MLP with output size all genres

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([generalized_mean_exponent])
    fully_connected_layer = lambda_layer(xception_output_feature_vector)
    fully_connected_layer = Dropout(0.05)(fully_connected_layer) # apply dropout to prevent overfitting
    fully_connected_layer = Activation('relu')(fully_connected_layer) # relu activation layer
    fully_connected_layer = Dense(num_labels, activation='softmax')(fully_connected_layer) # fully connected l

    top_model = Model(inputs=xception_output_feature_vector, outputs=fully_connected_layer)
    input_image_size = Input(list(input_shape) + [num_channels])

    model_connection = xception_model(input_image_size) # connect the input to the xception model
    model_connection = top_model(model_connection) # connect xception to the fully connected layer
    model = Model(inputs=input_image_size, outputs=model_connection)
    # model.summary()
    return model

def main():
    image_shape = (299, 299)
    num_channels = 3
    
    parser = argparse.ArgumentParser(description='XceptionNet model')
    parser.add_argument(
        'mode',
        choices=['train', 'test', 'predict', 'evaluate'],
        help="Mode in which the model should be run"
        )
    parser.add_argument(
        '-weights', '--weights',
        type=str,
        default="weights/xception_checkpoint-best.h5",
        help="If testing, path of saved weights"
        )
    parser.add_argument(
        '-loss', '--loss',
        type=str,
        default="binary_crossentropy",
        help="loss function to monitor. Default: binary_crossentropy"
        )
    parser.add_argument(
        '-samples', '--samples',
        type=int,
        default="3",
        help="if testing, number of samples to show and save (if desired). Default 3."
        )
    parser.add_argument(
        '-save', '--save',
        type=bool,
        default=False,
        help="if testing, boolean defining whether to save samples. Default False"
        )
    parser.add_argument(
        '-img_path', '--img_path',
        type=str,
        default="",
        help="if predicting, path to the image"
    )
    args = parser.parse_args()

    if args.mode == 'test':
        test_data_evaluation(args.weights, 'XceptionNet')
        # test(args.weights, num_samples=args.samples, save_imgs=args.save)
    elif args.mode == 'train':
        train()
        # evaluate(args.by, args.weightsfolder)
    elif args.mode == 'predict':
        predict(args.img_path)
    else:
        pass

if __name__ == "__main__":
    main()
    # load_data()
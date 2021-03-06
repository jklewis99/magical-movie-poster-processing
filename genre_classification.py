import os
import cv2
import glob
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xception_transfer as xception

from PIL import Image
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from utils.read_images import read_images
from utils.data_read import load_data, load_train_test
from utils.misc import labels_to_text, show_poster_and_genres, get_genres

parser = argparse.ArgumentParser(description="Classify movie genres based on the movie poster", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    'mode',
    choices=['train', 'predict', 'find_threshold', 'class_activation_map'],
    help="This is a REQUIRED PARAMETER to set mode to 'train' or 'predict' or 'find_threshold' or 'class_activation_map'")
parser.add_argument(
    '--model', '-model',
    choices=['1', 'NasNet', '2', 'InceptionResNet', '3', 'XceptionNet'],
    default='NasNet',
    help="This is a PARAMETER to specify which model to use. Default is 'NasNet':\
        \n'1' or 'NasNet' for NasNetLarge,\
        \n'2' or 'InceptionResNet' for InceptionResNetV2,\
        \n'3' or 'XceptionNet` for Xception")
parser.add_argument(
    '--path', '-path',
    default='',
    help="Set the path of the image to predict or to create class_activation_map.\
        \nOnly available in 'test' or 'class_activation_map' mode")
parser.add_argument(
    '--train_mode', '-train_mode',
    choices=[1, 2],
    default=1,
    type=int,
    help="Set the mode for training. Default train_mode is 1.\
        \n'1' for training a new model\
        \n'2' for training an existing model")

model_path = ''  # Path of the model. This variable is modified in the 'main' function
batch_size = 4  # Size of batch for neural network
patience = 10  # The number of times the model keeps on training before stopping when EarlyStopping conditions are satisfied
epochs = 100  # The number of times the neural network trains

learning_rate = 0.001  # The learning rate of the model
amsgrad = False  # This indicates whether or not to enable advanced algorithm
epsilon = 0.1  # This preventing the computation during the training from dividing by 0

height = 331  # Height of the image. The value should not be changed
width = 331  # Width of the image. The value should not be changed
channel = 3  # Channel of the image. 1 for grayscale and 3 for RGB

threshold = 0.4  # The threshold which the model classifies the poster as the specified genre. This value is obtained after training the model and run the function find_threshold()


def get_NasNetLarge(num_output):
    '''
    This function builds the neural network based on the NasNetLarge model

    Parameters
    ==========
    `num_output`:
        number of classes requiring detection

    Return
    ==========
    NasNetLarge model with an updated last layer MLP
    '''
    pre_train_model = NASNetLarge(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model

def get_InceptionResnetV2(num_output):
    '''
    This function builds the neural network based on the InceptionResNetV2 model

    Parameters
    ==========
    `num_output`:
        number of classes requiring detection

    Return
    ==========
    InceptionResNetV2 model with an updated last layer MLP
    '''
    pre_train_model = InceptionResNetV2(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model

def get_XceptionNet(num_output):
    '''
    This function builds the neural network based on the XceptionNet model

    Parameters
    ==========
    `num_output`:
        number of classes requiring detection

    Return
    ==========
    XceptionNet model with an updated last layer MLP
    '''
    pre_train_model = Xception(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model

def plot_history(result):
    '''
    This function plots history of model training

    Parameters
    ==========
    `result`:
        A History object. `result.history` is a record of training loss values 
        and metrics values at successive epochs, as well as validation loss values and 
        validation accuracy values. It is required that `result.history` contains 'accuracy'
        as a metric.
    '''
    # Create and save a plot consists of val_loss and val_acc
    fig = plt.figure()

    loss = fig.add_subplot(1, 1, 1)
    loss.plot(result.history['val_loss'], label='val_loss')

    acc = fig.add_subplot(2, 1, 1)
    acc.plot(result.history['val_accuracy'], label='val_accuracy')

    plt.savefig(os.path.join(os.getcwd(), 'Data', 'result.png'))

def train(train_mode, model_type, model_path):
    '''
    This function trains the neural network and plots the history of
    training.
    
    Parameters
    ==========
    `train_mode`:
        1 or 2, specifies whether to start training (1), or to continue 
        training (2) with model `model_path`

    `model_type`:
        specifies the model to train

    `model_path`:
        absolute or relative path to the model's file
    '''
    genres = get_genres()

    if train_mode == 2:
        model = get_model(model_path, model_type)

    else:
        if model_type in ['1', 'NasNet']:
            model = get_NasNetLarge(len(genres))
        elif model_type in ['2', 'InceptionResNet']:
            model = get_InceptionResnetV2(len(genres))
        else:
            #TODO: FIX MODEL
            # model = get_XceptionNet(len(genres))
            model = xception.build_model(len(genres))
    shape = get_model_shape(model_type)

    x_train, y_train, x_test, y_test, _ = load_train_test(img_shape=shape)  # Load data

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
        )
    # Train the model
    result = model.fit(x_train, y_train,
                       validation_data=(x_test, y_test),
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[checkpoint, EarlyStopping(monitor='val_loss', mode='min', patience=patience)]
                       )

    # Plot history
    plot_history(result)

def find_threshold(model_path, model_type):
    '''
    This function finds the threshold for predicting

    Parameters
    ==========
    `model_path`:
        absolute or relative path to the model's file

    `model_type`:
        string type of the model
    '''
    shape = get_model_shape(model_type)
    model = get_model(model_path, model_type)  # Load model
    x_train, y_train, x_test, y_test, genres = load_train_test(img_shape=shape)  # Load data
    data = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)
    thresholds = [0.3, 0.2, 0.1]
    plot_data = []
    perf_data = []

    for threshold in thresholds:
        genres_accuracy, perfect_accuracy = evaluate(model, threshold, data, label, len(genres))
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

def get_model(model_path, model_type):
    '''
    return model as specified by model_path. XceptionNet loads
    weights, while the other two models load models

    Parameters
    ==========
    `model_path`:
        absolute or relative path to the model's file

    `model_type`:
        string type of the model
    '''
    if model_type in ['3', 'XceptionNet']:
        model = xception.load_model(model_path, 25)
    else:
        model = load_model(model_path)  # Load model
    return model

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
    shape = get_model_shape(model_type)
    model = get_model(model_path, model_type)  # Load model
    x_test, y_test, _, genres = load_data(img_shape=shape)  # Load data
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

def predict(img_path, genres, model_path, model_type, actual=None):
    '''
    This function predicts a specific poster at specified `img_path`

    Parameters
    ==========
    `img_path`:
        absolute or relative path to poster image

    `genres`:
        all possible string genres for a film (as defined in training)
    
    `model_path`:
        absolute or relative path to the model's file

    Keyword Args
    ==========
    `actual`:
        ground truth labels
    '''
    result = []  # This variable refers to all the genre that the poster might belong to
    model = get_model(model_path, model_type) # Load model
    shape = get_model_shape(model_type)
    data = np.zeros((1, shape[0], shape[1], channel))  # Initialize data
    data[0] = np.asarray(Image.open(img_path).resize(shape, Image.ANTIALIAS))  # Load data
    prediction = model.predict(data)  # Make prediction

    # Get the genre that is equal or larger than the threshold
    for i in range(len(genres)):
        if prediction[0][i] >= threshold:
            result.append(genres[i])
    show_poster_and_genres(img_path, result, text_actual=actual)
    print('Genre(s): ' + str(result))

def class_activation_map(img_path, genres, model_type, model_path):
    '''
    This function overlays an activation map on the specified image

    Parameters
    ==========
    `img_path`:
        absolute or relative path to poster image

    `genres`:
        all possible string genres for a film (as defined in training)

    `model_type`:
        type of model used to predict the image

    `model_path`:
        absolute or relative path to model 
    '''
    if model_type not in ['1', 'NasNet']:
        raise NotImplementedError("\nOOPS! This function can only be called on the NasNet Model at the moment...\
            \nPlease change the --model parameter to '1' or 'NasNet'")
    tf.compat.v1.disable_eager_execution()  # This is used to solve problems for Tensorflow v2
    result = []  # This variable stores the genre predictions
    intensity = 0.8  # This is the transparency of the heat map

    shape = get_model_shape(model_type)
    # Load and process images
    img = image.load_img(img_path, target_size=shape)
    process_img = image.img_to_array(img)
    process_img = np.expand_dims(process_img, axis=0)

    result_img = [img]  # This variable stores the default image the class_activation_map for each genre

    model = get_model(model_path, model_type) # Load model
    predictions = model.predict(process_img)

    # Get the genre that is equal or larger than the threshold
    for prediction in predictions:
        index = 0

        for value in prediction:
            if value >= threshold:
                result.append(index)

            index += 1

    # Get the final convolution layer from each model.
    # The index is determined by the following code: 
    # dictionary = {v.name: i for i, v in enumerate(model.layers)}
    final_convolution_layer = {
        '1': 1038,
        'NasNet': 1038,
        '2': 149,
        'InceptionResNet': 149,
        '3': 131,
        'XceptionNet': 131 
    }
    layer_index = final_convolution_layer[model_type]
    layer = model.get_layer(index=layer_index)

    # Loop through each genre in the result
    for data in result:
        # Compute value to create heat map
        grads = K.gradients(model.output[:, data], layer.output)[0]
        poly_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [poly_grads, layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate(process_img)

        for i in range(np.shape(poly_grads)[0]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # Plot the heat map
        heat_map = np.mean(conv_layer_output_value, axis=-1)
        heat_map = np.maximum(heat_map, 0)
        heat_map /= np.max(heat_map)
        heat_map = cv2.resize(heat_map, (height, width))
        heat_map = np.uint8(heat_map * 255)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

        # Load image using opencv
        img_data = cv2.imread(img_path)
        img_data = cv2.resize(img_data, (height, width))

        # Overlay the image with the heat_map
        class_activation_map = cv2.cvtColor(
                                    cv2.addWeighted(heat_map, intensity, img_data, 1, 0),
                                    cv2.COLOR_BGR2RGB
                                    )
        result_img.append(class_activation_map)

    # Plot the result
    fig = plt.figure()
    plot_index = 1
    for img in result_img:
        ax = fig.add_subplot(1, len(result_img), plot_index)
        ax.imshow(img)

        if plot_index == 1:
            ax.set_title('Original Image')
        else:
            ax.set_title(genres[result[plot_index-2]])

        plot_index += 1
    plt.show()

def get_model_shape(model):
    '''
    get the needed input shape based on the model

    Parameters
    ==========
    `model`:
        string of that specifies the model

    Return
    shape

    tuple representing the input (height, width)
    '''
    input_shape_map = {
        '1': (331,331),
        'NasNet': (331,331),
        '2': (300,300),
        'InceptionResNet': (300,300),
        '3': (299,299),
        'XceptionNet': (299,299) 
    }
    return input_shape_map[model]

def main(args):
    '''
    Main function to define what functions to run
    '''
    mode = args.mode
    model_type = args.model

    if model_type in ['1', 'NasNet']:
        model_path = os.path.join(os.getcwd(), 'models', 'NasNetLarge.h5')
    elif model_type in ['2', 'InceptionResNet']:
        model_path = os.path.join(os.getcwd(), 'models', 'InceptionResNetV2.h5')
    else:
        model_path = os.path.join(os.getcwd(), 'models', 'XceptionNet.h5')

    if mode == 'train':
        train(args.train_mode, model_type, model_path)
    
    elif mode in ['predict','class_activation_map']:
        path = args.path
        genres = get_genres()

        if path == '':
            print('Please provide a path to the image')
        else:
            if mode == 'predict':
                predict(path, genres, model_path, model_type)
            else:
                class_activation_map(path, genres, model_type, model_path)
    elif mode == 'find_threshold':
        find_threshold(model_path, model_type)


if __name__ == '__main__':
    try:
        main(parser.parse_args())
    except Exception as e:
        print(e)

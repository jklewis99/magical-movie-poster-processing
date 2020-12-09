import os
import cv2
import glob
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from keras.layers import Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_resnet_v2 import InceptionResNetV2

parser = argparse.ArgumentParser(description="Classify movie genres based on the movie poster")
parser.add_argument('mode', choices=['train', 'test', 'find_threshold', 'class_activation_map'], help="This is a REQUIRED PARAMETER to set mode to 'train' or 'test' or 'find_threshold' or 'class_activation_map'")
parser.add_argument('--model', default='', help="This is a REQUIRED PARAMETER to choose the model: 1 for NasNetLarge, 2 for InceptionResNetV2, 3 for Xception")
parser.add_argument('--path', default='', help="Set the path of the image to predict or to create class_activation_map\nOnly available in 'test' or 'class_activation_map' mode")
parser.add_argument('--train_mode', default='1', help="Set the mode for training\n'1' for training a new model\n'2' for training an existing model\nDefault train_mode is 1")

model_path = ''  # Path of the model. This variable is modified in the 'main' function
genre = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', ' Sport', 'Thriller', 'War', 'Western']  # All the genres
validation_split = 0.2  # Ratio of data reserved for validation
batch_size = 4  # Size of batch for neural network
patience = 10  # The number of time the model keeps on training before stopping when EarlyStopping conditions is satisfied
epochs = 100  # The number of times the neural network trains

learning_rate = 0.001  # The learning rate of the model
amsgrad = False  # This indicates whether or not enable advanced algorithm
epsilon = 0.1  # This preventing the computation during the training from dividing by 0

height = 331  # Height of the image. The value should not be changed
width = 331  # Width of the image. The value should not be changed
channel = 3  # Channel of the image. 1 for grayscale and 3 for RGB

threshold = 0.6  # The threshold which the model classifies the poster as the specified genre. This value is obtained after training the model and run the function find_threshold()


# This function builds the neural network based on the NasNetLarge model
def NasNetLarge(num_output):
    pre_train_model = NASNetLarge(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model


# This function builds the neural network based on the InceptionResNetV2 model
def InceptionResnetV2(num_output):
    pre_train_model = InceptionResNetV2(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model


# This function builds the neural network based on the Xception model
def XceptionNet(num_output):
    pre_train_model = Xception(include_top=False, pooling='avg')
    output = Dense(num_output, activation='sigmoid')(pre_train_model.output)

    model = Model(inputs=pre_train_model.input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon, amsgrad=amsgrad), metrics=['accuracy'])
    print(model.summary())
    return model


# This function loads data
def load_data():
    # Initialize variables
    path = os.path.join(os.getcwd(), 'Data', 'Movie_Poster_Dataset', 'Movie_Poster_Dataset', '')
    sub_path = glob.glob(path + '*')
    label_path = os.path.join(os.getcwd(), 'Data', 'Labels.txt')
    img_path = []
    info = {}

    # Get the path of all images
    for data_path in sub_path:
        img_path.extend(glob.glob(os.path.join(data_path, '*')))

    # Shuffle the path to all images to randomize data
    np.random.shuffle(img_path)

    # Open the label file and add label to the dictionary. The key of the dictionary is the id of the image
    with open(label_path, 'r') as f:
        for line in f.readlines()[1:]:
            info[line.split(',')[0]] = line.split(']')[1].strip().split(',')[1:]

    # Initialize the variables
    x_train = np.zeros((round(len(info) * (1 - validation_split)), height, width, channel))
    y_train = np.zeros((round(len(info) * (1 - validation_split)), len(genre)))
    x_test = np.zeros((round(len(info) * validation_split), height, width, channel))
    y_test = np.zeros((round(len(info) * validation_split), len(genre)))

    # Load training and testing data
    for i in range(len(info)):
        if img_path[i].split('\\')[-1].split('.')[0] in list(info.keys()):
            if i < round(len(info) * (1 - validation_split)):
                x_train[i] = np.asarray(Image.open(img_path[i]).resize((height, width), Image.ANTIALIAS))
                y_train[i] = np.asarray(info[img_path[i].split('\\')[-1].split('.')[0]])
            else:
                x_test[i - len(x_train)] = np.asarray(Image.open(img_path[i]).resize((height, width), Image.ANTIALIAS))
                y_test[i - len(y_train)] = np.asarray(info[img_path[i].split('\\')[-1].split('.')[0]])

    return x_train, y_train, x_test, y_test


# This function plots history of model training
def plot_history(result):
    # Create and save a plot consists of val_loss and val_acc
    fig = plt.figure()

    loss = fig.add_subplot(1, 1, 1)
    loss.plot(result.history['val_loss'], label='val_loss')

    acc = fig.add_subplot(2, 1, 1)
    acc.plot(result.history['val_accuracy'], label='val_accuracy')

    plt.savefig(os.path.join(os.getcwd(), 'Data', 'result.png'))


# This function trains the neural network
def train(mode, model_type):
    x_train, y_train, x_test, y_test = load_data()  # Load data

    if mode == '2':
        model = load_model(model_path)
    else:
        if model_type == '1':
            model = NasNetLarge(len(genre))
        elif model_type == '2':
            model = InceptionResnetV2(len(genre))
        else:
            model = XceptionNet(len(genre))

    # Train the model
    result = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                       callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), EarlyStopping(monitor='val_loss', mode='min', patience=patience)])

    # Plot history
    plot_history(result)


# This function finds the threshold for predicting
def find_threshold():
    model = load_model(model_path)  # Load model
    model_name = str(model_path.split('\\')[-1]).split('.')[0]
    x_train, y_train, x_test, y_test = load_data()  # Load data
    data = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)
    thresholds = [0.3, 0.2, 0.1]
    plot_data = []

    start = time.time()
    for threshold in thresholds:
        plot_data.append(evaluate(model, threshold, data, label, len(genre)))
        print(plot_data)
    end = time.time()

    fig = plt.figure()
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(thresholds, plot_data)
    graph.subtitle(model_name + '\n#Params: ' + model.count_params() + '\nEvaluation Completion Time: ' + str(int(end - start)) + ' seconds')
    graph.xlabel('Threshold')
    graph.ylabel('Accuracy')
    plt.savefig(os.path.join(os.getcwd(), 'Data', model_name + '_evaluation.png'))
    plt.show()


# This function evaluates the model performance given the threshold and other data
def evaluate(model, threshold, data, label, num):
    prediction = model.predict(data)
    acc = 0

    for result, sub_label in zip(prediction, label):
        for i in range(num):
            if result[i] >= threshold and sub_label[i] == 1:
                acc += 1

            if result[i] < threshold and sub_label[i] == 0:
                acc += 1

    return acc / (len(label) * num)


# This function predict the poster
def predict(path):
    result = []  # This variable refers to all the genre that the poster might belong to
    model = load_model(model_path)  # Load model
    data = np.zeros((1, height, width, channel))  # Initialize data
    data[0] = np.asarray(Image.open(path).resize((height, width), Image.ANTIALIAS))  # Load data
    prediction = model.predict(data)  # Make prediction

    # Get the genre that is equal or larger than the threshold
    for i in range(len(genre)):
        if prediction[0][i] >= threshold:
            result.append(genre[i])

    print('Genre: ' + str(result))


# This function create the class_activation_map
def class_activation_map(img_path):
    tf.compat.v1.disable_eager_execution()  # This is used to solve problems for Tensorflow v2
    result = []  # This variable stores the genre predictions
    intensity = 0.8  # This is the transparency of the heat map

    # Load and process images
    img = image.load_img(img_path, target_size=(height, width))
    process_img = image.img_to_array(img)
    process_img = np.expand_dims(process_img, axis=0)

    result_img = [img]  # This variable stores the default image the class_activation_map for each genre

    # Load model and predict the result
    model = load_model(model_path)
    predictions = model.predict(process_img)

    # Get the genre that is equal or larger than the threshold
    for prediction in predictions:
        index = 0

        for value in prediction:
            if value >= threshold:
                result.append(index)

            index += 1

    # Get the final convolution layer. The index is determine by the following code: dictionary = {v.name: i for i, v in enumerate(model.layers)}. Change this fro different model
    layer = model.get_layer(index=1038)

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
        class_activation_map = cv2.cvtColor(cv2.addWeighted(heat_map, intensity, img_data, 1, 0), cv2.COLOR_BGR2RGB)
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
            ax.set_title(genre[result[plot_index-2]])

        plot_index += 1
    plt.show()


# This is a main function
def main(args):
    mode = args.mode
    model_type = args.model

    if model_type == '':
        print('Please provide the type of the model')
    else:
        global model_path

        if model_type == '1':
            model_path = os.path.join(os.getcwd(), 'NasNetLarge.h5')
        elif model_type == '2':
            model_path = os.path.join(os.getcwd(), 'InceptionResNetV2.h5')
        else:
            model_path = os.path.join(os.getcwd(), 'XceptionNet.h5')

        if mode == 'train':
            valid_mode = ['1', '2', '3', '4']
            train_mode = args.train_mode

            if train_mode in valid_mode:
                train(train_mode, model_type)
            else:
                print('Invalid Mode')
        elif mode == 'predict':
            path = args.path

            if path == '':
                print('Please provide a path to the image')
            else:
                predict(path)
        elif mode == 'find_threshold':
            find_threshold()
        elif mode == 'class_activation_map':
            path = args.path

            if path == '':
                print('Please provide a path to the image')
            else:
                class_activation_map(path)
        else:
            print('Invalid Mode')


if __name__ == '__main__':
    try:
        main(parser.parse_args())
    except Exception as e:
        print(e)

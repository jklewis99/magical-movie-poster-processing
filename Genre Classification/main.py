import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, Input, GlobalAveragePooling2D, Dropout, Dense

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code
parser = argparse.ArgumentParser(description="Test a perceptron to classify letters.")
parser.add_argument('mode', choices=['train', 'test', 'find_threshold'], help="set mode to 'train' or 'test' or 'find_threshold'")
parser.add_argument('--path', default='', help="Set the path of the image to predict\nOnly available in 'test' mode")

model_path = os.path.join(os.getcwd(), 'model.h5')  # Path of the model
genre = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', ' Sport', 'Thriller', 'War', 'Western']
validation_split = 0.2  # Ratio of data reserved for validation
batch_size = 16  # Size of batch for neural network
patience = 10  # The number of time the model keeps on training before stopping when EarlyStopping conditions is satisfied
epochs = 100  # The number of times the neural network trains

height = 300  # Height of the image
width = 300  # Width of the image
channel = 3  # Channel of the image. 1 for grayscale and 3 for RGB

threshold = 0.6  # The threshold which the model classifies the poster as the specified genre. This value is obtained after training the model and run the function find_threshold()


# Stem module of the Inception_ResNet_v2
def stem(x):
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2])

    x3 = Conv2D(64, kernel_size=(1, 1), padding='same')(x)
    x3 = Conv2D(96, kernel_size=(3, 3), padding='valid')(x3)

    x4 = Conv2D(64, kernel_size=(1, 1), padding='same')(x)
    x4 = Conv2D(64, kernel_size=(7, 3), padding='same')(x4)
    x4 = Conv2D(64, kernel_size=(1, 7), padding='same')(x4)
    x4 = Conv2D(96, kernel_size=(3, 3), padding='valid')(x4)

    x = Concatenate(axis=3)([x3, x4])

    x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x6 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x5, x6])
    x = Activation('relu')(x)
    return x


# InceptionA module of the Inception_ResNet_v2
def InceptionA(x):
    x1 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(32, kernel_size=(3, 3), padding='same')(x2)

    x3 = Conv2D(32, kernel_size=(1, 1), padding='same')(x)
    x3 = Conv2D(48, kernel_size=(3, 3), padding='same')(x3)
    x3 = Conv2D(64, kernel_size=(3, 3), padding='same')(x3)

    x4 = Concatenate(axis=3)([x1, x2, x3])
    x4 = Conv2D(384, kernel_size=(1, 1), padding='same')(x4)

    inception_a = Concatenate(axis=3)([x, x4])
    inception_a = Activation('relu')(inception_a)

    return inception_a


# InceptionB module of the Inception_ResNet_v2
def InceptionB(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(128, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(160, kernel_size=(1, 7), padding='same')(x2)
    x2 = Conv2D(192, kernel_size=(7, 1), padding='same')(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(1154, kernel_size=(1, 1), padding='same')(x3)

    inception_b = Concatenate(axis=3)([x, x3])
    inception_b = Activation('relu')(inception_b)

    return inception_b


# InceptionC module of the Inception_ResNet_v2
def InceptionC(x):
    x1 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)

    x2 = Conv2D(192, kernel_size=(1, 1), padding='same')(x)
    x2 = Conv2D(224, kernel_size=(1, 3), padding='same')(x2)
    x2 = Conv2D(256, kernel_size=(3, 1), padding='same')(x2)

    x3 = Concatenate(axis=3)([x1, x2])
    x3 = Conv2D(2048, kernel_size=(1, 1), padding='same')(x3)

    inception_c = Concatenate(axis=3)([x, x3])
    inception_c = Activation('relu')(inception_c)

    return inception_c


# ReductionA module of the Inception_ResNet_v2
def ReductionA(x):
    x1 = Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = Conv2D(224, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x1)

    x2 = Conv2D(192, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3])
    x = Activation('relu')(x)
    return x


# ReductionB module of the Inception_ResNet_v2
def ReductionB(x):
    x1 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x1)

    x2 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x2)

    x3 = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x3 = Conv2D(384, kernel_size=(3, 3), strides=(2, 2), padding='valid')(x3)

    x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate(axis=3)([x1, x2, x3, x4])
    x = Activation('relu')(x)
    return x


# Output of neural network
def output_networks(x, num_output):
    x1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = Activation('relu')(x1)

    x2 = Concatenate(axis=3)([x, x1])
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dropout(rate=0.25)(x2)

    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(32, activation='relu')(x2)
    output = Dense(num_output, activation='sigmoid')(x2)

    return output


# This function build the neural network based on the Inception_ResNet_v2
def InceptionResnetV2(num_output):
    input_data = Input((height, width, channel))  # Input data
    input_layer = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_data)

    # The structure follows the structure of the Inception_ResNet_v2 with customized output layer
    x = stem(input_layer)

    for i in range(4):
        x = InceptionA(x)

    x = ReductionA(x)

    for i in range(6):
        x = InceptionB(x)

    x = ReductionB(x)

    for i in range(3):
        x = InceptionC(x)

    outputs = output_networks(x, num_output)

    # Build and compile the model
    model = Model(inputs=input_data, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # This is used for analyzing the structure of the model
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
def train():
    x_train, y_train, x_test, y_test = load_data()  # Load data
    model = InceptionResnetV2(len(genre))  # Build neural network

    # Train the model
    result = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                       callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), EarlyStopping(monitor='val_loss', mode='min', patience=patience)])

    # Plot history
    plot_history(result)


# This function finds the threshold for predicting
def find_threshold():
    model = load_model(model_path)  # Load model
    x_train, y_train, x_test, y_test = load_data()  # Load data
    data = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    plot_data = []

    for threshold in thresholds:
        plot_data.append(evaluate(model, threshold, data, label, len(genre)))

    fig = plt.figure()
    graph = fig.add_subplot(1, 1, 1)
    graph.plot(thresholds, plot_data)
    plt.savefig(os.path.join(os.getcwd(), 'Data', 'evaluation.png'))
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

    for i in range(len(genre)):
        if prediction[0][i] >= threshold:
            result.append(genre[i])

    print('Genre: ' + str(result))


# This is a main function
def main(args):
    mode = args.mode

    if mode == 'train':
        train()
    elif mode == 'predict':
        path = args.path

        if path == '':
            print('Please provide a path to the image')
        else:
            predict(path)
    elif mode == 'find_threshold':
        find_threshold()
    else:
        print('Invalid Mode')


if __name__ == '__main__':
    try:
        main(parser.parse_args())
    except Exception as e:
        print(e)

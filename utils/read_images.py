import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def read_images(images_list, with_tensorflow=False, path=f"data/Images", \
    has_extension=False, extension_type="jpg"):
    '''
    takes a list of file names without extension, reads the images, 
    and returns a numpy array of the images

    Parameters
    ==========
    `images_list`:
        list of strings
    
    Keyword Arguments
    ==========
    `with_tensorflow`:
        determines which child function to call
    
    `path`:
        relative path to the image files
    
    `has_extension`:
        if set to True, does not add `extension_type` to the strings in `images_list`
    
    `extension_type`:
        only read if `has_extension` is set to False. Adds the string `extension_type`
        to each string in `images_list`

    Returns
    ==========
    numpy array of the read images
    '''
    if not has_extension:
        images_path_list = [os.path.join(path, f"{image}.{extension_type}") for image in images_list]
    else:
        images_path_list = [os.path.join(path, image) for image in images_list]
    
    if with_tensorflow:
        images_read = read_images_tensorflow(images_path_list, (350, 350, 3))
    else:
        images_read = read_images_opencv(images_path_list, (240, 360))

    return images_read

def read_images_opencv(images_list, dimensions):
    '''
    takes a list of paths to images and reads them with opencv

    Parameters
    ==========
    `images_list`:
        list of strings

    `dimensions`:
        tuple specifying width and height of image

    Returns
    ==========
    numpy array of the read images
    '''
    
    dims = set()

    np_images = []
    for img_path in images_list:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
        dims.add(img.shape)
        np_images.append(img)
    print(dims)

    return np.array(np_images)

def read_images_tensorflow(images_list, dimensions):
    '''
    takes a list of paths to images and reads them with tensorflow's image module

    Parameters
    ==========
    `images_list`:
        list of strings

    `dimensions`:
        tuple specifying width and height and channels of image

    Returns
    ==========
    numpy array of the read images
    '''

    np_images = []
    for img_path in images_list:
        img = image.load_img(img_path, target_size=dimensions)
        img = image.img_to_array(img)
        np_images.append(img)

    return np.stack(np_images)
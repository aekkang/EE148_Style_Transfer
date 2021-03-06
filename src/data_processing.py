######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     data_processing.py
# Desc:     Defines data processing functions for style transfer.
######################################################################

import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from defaults import *


##############################
# PARAMETERS
##############################

# ImageNet processing parameters.
IMAGENET_MEAN = [103.939, 116.779, 123.68]


##############################
# PREPROCESSING
##############################

def preprocess_img(path, width, height):
    """Preprocess a single image at the given path into the network input format.
    """

    # Load image.
    img = load_img(path, target_size=(height, width))

    # Convert the image to a dataset.
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    # Use the network preprocess method.
    arr = NETWORK_PREPROCESS(arr)

    # Contain the dataset in a Keras variable.
    return arr


##############################
# DEPROCESSING
##############################

def deprocess_img(img, width, height):
    """Deprocess the given image to a consumable format.
    """

    # Reshape the image.
    img = img.reshape((height, width, 3))

    # Undo the network preprocess method.
    for i in range(len(IMAGENET_MEAN)):
        img[:,:,i] += IMAGENET_MEAN[i]

    # Clip the numbers to the appropriate range.
    img = np.clip(img, 0, 255).astype('uint8')

    # Swap channel order from RGB to BGR.
    return img[:, :, ::-1]

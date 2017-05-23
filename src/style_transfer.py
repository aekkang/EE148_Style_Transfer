######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     style_transfer.py
# Desc:     Contains script for style transfer.
######################################################################

import argparse
import numpy as np
from keras import backend as K
from scipy.optimize import minimize

from data_processing import *
from loss import *
from config import *


##############################
# DATA PROCESSING
##############################

# Parse script arguments.
parser = argparse.ArgumentParser(description="Style transfer using neural networks.")
parser.add_argument("content_path", help="Path to the content image.")
parser.add_argument("style_path", help="Path to the style image.")
parser.add_argument("combination_path", help="Desired path to the combined image.")

args = parser.parse_args()
content_path = args.content_path
style_path = args.style_path

# Calculate desired width and height.
width, height = load_img(content_path).size

# Load images and declare variable to store the combined image.
content = preprocess_img(content_path, height, width)
style = preprocess_img(style_path, height, width)
combination = K.placeholder((1, height, width, 3))

# Concatenate the images into one tensor.
input_tensor = K.concatenate((content, style, combination), axis=0)


##############################
# MODEL ARCHITECTURE
##############################

# Load the pre-trained network.
model = NETWORK_MODEL(input_tensor=input_tensor, include_top=False)

# Calculate the total loss and its gradients with respect to the
# combined image.
loss = total_loss(model)
gradients = K.gradients(loss, combination)

# Function to minimize.
f_loss = K.function([combination], [loss])
f_gradients = K.function([combination], [gradients])


##############################
# IMAGE SEARCH
##############################

# Start with a white noise image.
combination_i = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

for i in range(ITERATIONS):
    print("Iteration: " + str(i))

    result = minimize(f_loss, combination_i, jac=f_gradients)
    print(result.status)
    print("Iteration loss: " + result.status)
    
    # Save iteration results.
    imsave(combination_path, deprocess_img(combination_i))

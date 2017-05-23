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
from cv2 import imwrite

from config import *
from data_processing import *
from loss import *
from minimizer import *


##############################
# ARGUMENT PROCESSING
##############################

# Parse script arguments.
parser = argparse.ArgumentParser(description="Style transfer using neural networks.")
parser.add_argument("style_path", help="Path to the style image.")

args = parser.parse_args()
dir_path = args.dir_path

if dir_path[-1] != '/':
    dir_path += '/'

content_path = dir_path + "content.jpg"
style_path = dir_path + "style.jpg"
combination_prefix = dir_path + "combination"


##############################
# DATA PROCESSING
##############################

# Calculate desired width and height.
width, height = load_img(content_path).size
width, height = int(HEIGHT * width / height), HEIGHT

# Load images and declare variable to store the combined image.
content = preprocess_img(content_path, width, height)
style = preprocess_img(style_path, width, height)
combination = K.placeholder((1, height, width, 3))

# Concatenate the images into one tensor.
input_tensor = K.concatenate((content, style, combination), axis=0)


##############################
# INITIALIZE MINIMIZER
##############################

# Load the pre-trained network.
model = NETWORK_MODEL(input_tensor=input_tensor, include_top=False)

# Calculate the total loss and its gradients with respect to the combined image.
loss = total_loss(model)
gradients = K.gradients(loss, combination)

# Function to minimize.
f_to_minimize = K.function([combination], [loss] + gradients)
minimizer = Minimizer(f_to_minimize, width, height, combination_prefix)


##############################
# RUN MINIMIZER
##############################

# Start with a white noise image.
combination_i = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
combination_i = combination_i.flatten()

# Minimize the loss with respect to the combined image. We use L-BFGS-B
# as the minimization method as it can work around memory constraints.
result = minimize(minimizer.f_loss, combination_i, jac=minimizer.f_gradients,
                  method="L-BFGS-B", callback=minimizer.write,
                  options={"maxiter": ITERS})

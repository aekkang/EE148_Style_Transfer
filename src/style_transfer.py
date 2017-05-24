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

from config import *
from data_processing import *
from loss import *
from minimizer import *


##############################
# ARGUMENT PROCESSING
##############################

# Parse script arguments.
parser = argparse.ArgumentParser(description="Style transfer using neural networks.")
parser.add_argument("dir_path", type=str, help="Path to the directory containing images.")
parser.add_argument("--content_weight", type=float, help="Weight on content in combined image.")
parser.add_argument("--style_weight", type=float, help="Weight on style in combined image.")
parser.add_argument("--variation_weight", type=float, help="Weight on variation in combined image.")
args = parser.parse_args()

# Directory variables.
dir_path = args.dir_path

if dir_path[-1] != '/':
    dir_path += '/'

content_path = dir_path + "content.jpg"
style_path = dir_path + "style.jpg"

# Weight terms.
content_weight = args.content_weight if args.content_weight is not None else CONTENT_WEIGHT
style_weight = args.style_weight if args.style_weight is not None else STYLE_WEIGHT
variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT


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
loss = total_loss(model, content_weight, style_weight, variation_weight)
gradients = K.gradients(loss, combination)

# Function to minimize.
f_to_minimize = K.function([combination], [loss] + gradients)
minimizer = Minimizer(f_to_minimize, width, height, dir_path)


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

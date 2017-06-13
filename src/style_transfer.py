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

from argument_processing import *
from data_processing import *
from loss import *
from minimizer import *


##############################
# ARGUMENT PARSING
##############################

# Parse script arguments.
parser = argparse.ArgumentParser(description="Style transfer using neural networks.")

# Path arguments.
parser.add_argument("input_dir", type=str, help="Path to the directory containing images.")

# All following arguments are optional.
# Weight arguments.
parser.add_argument("--content_weight", type=float, help="Weight on content in combined image.")
parser.add_argument("--style_weight", type=float, help="Weight on style in combined image.")
parser.add_argument("--style_ratios", type=float, nargs='*', help="Ratio of style weights in combined image.")
parser.add_argument("--variation_weight", type=float, help="Weight on variation in combined image.")
parser.add_argument("--style_layer_weights", type=float, nargs='*', help="Weights of each style layer.")

# Saving & loading arguments.
parser.add_argument("--load_previous", action="store_true", help="Start current minimization from results of previous minimization.")
parser.add_argument("--save_per_n_iters", type=int, help="Number of iterations to run the minimizations before each save.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite directory.")
parser.add_argument("--start_state", type=str, help="State to start from.")

# Miscellaneous arguments.
parser.add_argument("--height", type=int, help="Height of combined image.")
parser.add_argument("--iters", type=int, help="Number of total iterations to run the minimization.")

args = parser.parse_args()

# Process arguments.
content_weight, style_weight, style_ratios, variation_weight, style_layer_weights, \
load_previous, start_state, save_per_n_iters, height, iters, latest_save_num, \
input_dir, output_dir, content_path, style_paths = process_args(args)


##############################
# DATA PROCESSING
##############################

# Calculate desired width and height.
w, h = load_img(content_path).size
width = int(height * w / h)

# Load images and declare variable to store the combined image.
content = K.variable(preprocess_img(content_path, width, height))
styles = []
for style_path in style_paths:
    styles.append(K.variable(preprocess_img(style_path, width, height)))
combination = K.placeholder((1, height, width, 3))

# Concatenate the images into one tensor.
input_tensor = K.concatenate([content] + styles + [combination], axis=0)


##############################
# INITIALIZE MINIMIZER
##############################

# Load the pre-trained network.
model = NETWORK_MODEL(input_tensor=input_tensor, include_top=False)

# Calculate the total loss and its gradients with respect to the combined image.
loss = total_loss(model, content_weight, style_weight, style_ratios, variation_weight, style_layer_weights)
gradients = K.gradients(loss, combination)

# Function to minimize.
f_to_minimize = K.function([combination], [loss] + gradients)
minimizer = Minimizer(f_to_minimize, width, height, iters, save_per_n_iters, output_dir, latest_save_num=latest_save_num)


##############################
# RUN MINIMIZER
##############################

if load_previous:
    # Start with the latest save.
    combination_i = preprocess_img(content_path, width, height)
elif start_state == "content":
    # Start from the content image.
    combination_i = np.copy(preprocess_img(content_path, width, height))
elif start_state == "blank":
    # Start from a blank image.
    combination_i = np.zeros((1, height, width, 3))
else:
    # Start with a white noise image.
    combination_i = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

combination_i = combination_i.flatten()

# Minimize the loss with respect to the combined image. We use L-BFGS-B
# as the minimization method as it can work around memory constraints.
result = minimize(minimizer.f_loss, combination_i, jac=minimizer.f_gradients,
                  method="L-BFGS-B", callback=minimizer.write,
                  options={"maxiter": iters})

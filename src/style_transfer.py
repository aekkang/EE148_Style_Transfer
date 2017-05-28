######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     style_transfer.py
# Desc:     Contains script for style transfer.
######################################################################

import argparse
import os
from glob import glob

import numpy as np
from keras import backend as K
from scipy.optimize import minimize

from defaults import *
from utility import *
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
parser.add_argument("--variation_weight", type=float, help="Weight on variation in combined image.")
parser.add_argument("--style_layers_weights", type=float, nargs='*', help="Weights of each layer.")

# Saving & loading arguments.
parser.add_argument("--load_previous", type=bool, help="Start current minimization from results of previous minimization.")
parser.add_argument("--save_per_n_iters", type=int, help="Number of iterations to run the minimizations before each save.")

# Miscellaneous arguments.
parser.add_argument("--height", type=int, help="Height of combined image.")
parser.add_argument("--iters", type=int, help="Number of total iterations to run the minimization.")

args = parser.parse_args()


##############################
# ARGUMENT PROCESSING
##############################

# Path arguments.
input_dir = os.path.abspath(args.input_dir)

# Weight arguments.
content_weight = args.content_weight if args.content_weight is not None else CONTENT_WEIGHT
style_weight = args.style_weight if args.style_weight is not None else STYLE_WEIGHT
variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT
style_layers_weights = args.style_layers_weights if args.style_layers_weights is not None else STYLE_LAYERS_WEIGHTS

if len(style_layers_weights) == len(STYLE_LAYERS):
    style_layers_weights = [float(w) / sum(style_layers_weights) for w in style_layers_weights]
else:
    raise ValueError("Number of weights must match number of layers.")

# Saving & loading arguments.
load_previous = args.load_previous if args.load_previous is not None else LOAD_PREVIOUS
save_per_n_iters = args.save_per_n_iters or SAVE_PER_N_ITERS

# Miscellaneous arguments.
height = args.height if args.height is not None else HEIGHT
iters = args.iters if args.iters is not None else ITERS

# Construct path to output directory.
output_dir = os.path.join(input_dir, "../{}_c{}_s{}_v{}_w".format(input_dir.split('/')[-1],
                            format_parameter(content_weight),
                            format_parameter(style_weight),
                            format_parameter(variation_weight)))

for weight in style_layers_weights:
    output_dir += "{:g}_".format(format_parameter(weight))

output_dir += "h{:d}".format(height)

output_dir = os.path.abspath(output_dir)

# Make output directory.
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Construct paths to input files.
if load_previous:
    # Find the latest save file.
    saves = glob(os.path.join(output_dir, "combination_*"))
    saves_nums = [int(previous.split('_')[-1]) for save in previous_saves]
    latest_save = previous_saves[np.argmax(save_nums)]
    latest_save_num = max(saves_nums)
    content_path = latest_save
else:
    latest_save_num = None
    content_path = os.path.join(input_dir, "content.jpg")

style_path = os.path.join(input_dir, "style.jpg")


##############################
# DATA PROCESSING
##############################

# Calculate desired width and height.
w, h = load_img(content_path).size
width = int(height * w / h)

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
loss = total_loss(model, content_weight, style_weight, variation_weight, style_layers_weights)
gradients = K.gradients(loss, combination)

# Function to minimize.
f_to_minimize = K.function([combination], [loss] + gradients)
minimizer = Minimizer(f_to_minimize, width, height, iters, save_per_n_iters, output_dir, load_previous=latest_save_num)


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

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
from scipy.optimize import fmin_l_bfgs_b

from cv2 import imwrite

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
combination_path = args.combination_path

# Calculate desired width and height.
width, height = load_img(content_path).size
width, height = int(HEIGHT * width / height), HEIGHT

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
# f_loss_helper = K.function([combination], [loss])
# f_gradients_helper = K.function([combination], gradients)

# def f_loss(combination_i):
#     combination_i = combination_i.reshape((1, height, width, 3))
#     return f_loss_helper([combination_i])[0]

# def f_gradients(combination_i):
#     combination_i = combination_i.reshape((1, height, width, 3))
#     return f_gradients_helper([combination_i])[0].flatten()

f_to_minimize = K.function([combination], [loss] + gradients)

class Minimizer(object):
    def __init__(self):
        self.loss = None
        self.gradients = None

    def f_loss(self, combination_i):
        combination_i = combination_i.reshape((1, height, width, 3))
        output = f_to_minimize([combination_i])
        self.loss = output[0]

        if len(output[1:]) == 1:
            self.gradients = output[1].flatten().astype('float64')
        else:
            self.gradients = np.array(output[1:]).flatten().astype('float64')

        return self.loss

    def f_gradients(self, combination_i):
        return np.copy(self.gradients).flatten()
        # combination_i = combination_i.reshape((1, height, width, 3))
        # return f_gradients_helper([combination_i])[0].flatten()


##############################
# IMAGE SEARCH
##############################

minimizer = Minimizer()

# Start with a white noise image.
combination_i = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
import time
start = time.time()
for i in range(ITERATIONS):
    print("Iteration: " + str(i))

    result = minimize(minimizer.f_loss, combination_i.flatten(), jac=minimizer.f_gradients, method="L-BFGS-B", options={"maxiter":20}, callback=lambda x:imwrite("../examples/combination_test.jpg", deprocess_img(x, height, width)))
    #combination_i, min_val, info = fmin_l_bfgs_b(minimizer.f_loss, combination_i.flatten(), fprime=minimizer.f_gradients, maxfun=20)
    #result = minimize(minimizer.f_loss, combination_i.flatten(), jac=minimizer.f_gradients)
    print(combination_i)
    print("Iteration loss: " + str(result.status))
    print("TIME: " + str(time.time() - start))    

    # Save iteration results.
    imwrite(combination_path, deprocess_img(combination_i, height, width))

######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     config.py
# Desc:     Sets parameters for style transfer.
######################################################################

from keras.applications import vgg19


##############################
# PARAMETERS
##############################

# ImageNet processing parameters.
IMAGENET_MEAN = [103.939, 116.779, 123.68]

# Network to use.
NETWORK_PREPROCESS = vgg19.preprocess_input
NETWORK_MODEL = vgg19.VGG19

# Layers to use.
CONTENT_LAYER = "block4_conv2"
STYLE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

# Weight terms.
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1e0
STYLE_LAYERS_WEIGHTS = [1. / len(STYLE_LAYERS) for _ in range(len(STYLE_LAYERS))]
VARIATION_WEIGHT = 1e0

# Miscellaneous parameters.
ITERATIONS = 10
HEIGHT = 250

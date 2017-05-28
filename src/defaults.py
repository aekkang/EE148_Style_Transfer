######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     defaults.py
# Desc:     Sets defaults parameters for style transfer.
######################################################################

from keras.applications import vgg19


##############################
# PARAMETERS
##############################

# Network to use.
NETWORK_PREPROCESS = vgg19.preprocess_input
NETWORK_MODEL = vgg19.VGG19

# Layers to use.
CONTENT_LAYER = "block4_conv2"
STYLE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

# Weight parameters.
CONTENT_WEIGHT = 1e0
STYLE_WEIGHT = 1e4
VARIATION_WEIGHT = 0
STYLE_LAYERS_WEIGHTS = [1. / len(STYLE_LAYERS) for _ in range(len(STYLE_LAYERS))]

# Saving & loading parameters.
LOAD_PREVIOUS = False
SAVE_PER_N_ITERS = 500

# Miscellaneous parameters.
HEIGHT = 400
ITERS = 2000

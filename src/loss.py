######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     loss.py
# Desc:     Defines loss functions for style transfer.
######################################################################

import numpy as np
from keras import backend as K

from defaults import *


##############################
# CONTENT LOSS
##############################

def content_loss(content_tensor, combination_tensor):
    """
    Given a layer output tensor of the combination image and the
    corresponding layer output tensor of the content image, evaluates
    the content loss.
    """

    return K.sum(K.square(content_tensor - combination_tensor)) / 2


##############################
# STYLE LOSS
##############################

def gram_matrix(tensor):
    """
    Given a layer output tensor, calculate its correponding
    Gram matrix tensor.
    """

    # Make the filter axis the primary axis, then convert the
    # n-dimensional tensor to a matrix tensor.
    tensor_flat = K.batch_flatten(K.permute_dimensions(tensor, (2, 0, 1)))

    return K.dot(tensor_flat, K.transpose(tensor_flat))

def style_loss(style_tensor, combination_tensor):
    """
    Given a layer output tensor of the combination image and the
    corresponding layer output tensor of the style image, evaluates
    the style loss.
    """

    # Calculate the weight corresponding to the layer.
    layer_shape = np.product(combination_tensor.get_shape().as_list())
    layer_weight = 1. / (4 * layer_shape ** 2)

    # Calculate the Gram matrices.
    style_gram = gram_matrix(style_tensor)
    combination_gram = gram_matrix(combination_tensor)

    return layer_weight * K.sum(K.square(style_gram - combination_gram))


##############################
# VARIATION LOSS
##############################

def variation_loss(img_tensor):
    """
    Given an image tensor, finds the variation throughout the image.
    """

    # Compute the vertical and horizontal pair-wise pixel variation.
    vertical_var = K.square(img_tensor[:, :-1, :-1] - img_tensor[:, 1:, :-1])
    horizontal_var = K.square(img_tensor[:, :-1, :-1] - img_tensor[:, -1:, :1])

    return K.sum(K.pow(vertical_var + horizontal_var, 1.25))


##############################
# TOTAL LOSS
##############################

def total_loss(model, content_weight, style_weight, style_ratios, variation_weight, style_layer_weights):
    """
    Given a model, calculate the total loss, which consists of
    the content, style, and variation losses.
    """

    n_styles = len(style_ratios)
    loss = K.variable(0.)

    # Content loss.
    content_tensor = model.get_layer(CONTENT_LAYER).output[0]
    combination_tensor = model.get_layer(CONTENT_LAYER).output[n_styles + 1]
    loss += content_weight * content_loss(content_tensor, combination_tensor)

    # Style loss.
    for i in range(n_styles):
        for j, style_layer in enumerate(STYLE_LAYERS):
            style_tensor = model.get_layer(style_layer).output[i + 1]
            combination_tensor = model.get_layer(style_layer).output[n_styles + 1]
            loss += style_weight * style_ratios[i] * style_layer_weights[j] \
                    * style_loss(style_tensor, combination_tensor)

    # Variation loss.
    loss += variation_weight * variation_loss(model.inputs[0][2])

    return loss

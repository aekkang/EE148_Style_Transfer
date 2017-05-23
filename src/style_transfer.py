######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     style_transfer.py
# Desc:     Contains the actual neural network necessary for the
#           style transfer.
#           Adapted from https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py.
######################################################################

import numpy as np
from keras.applications.vgg19 import VGG19
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data_processing import *
from utility import *


##############################
# DATA PREPROCESSING
##############################

# # Load the dataset.
# (X_train, Y_train), (X_test, Y_test) = load_data()

# # Reshape the dataset to the desired format.
# loc_train = Y_train.reshape(Y_train.shape[0], 1, 1, 4)
# conf_train = np.ones((Y_train.shape[0], 1, 1, 1))
# Y_train = np.concatenate((loc_train, conf_train), axis=3)
# Y_train = transform(Y_train)

# loc_test = Y_test.reshape(Y_test.shape[0], 1, 1, 4)
# conf_test = np.ones((Y_test.shape[0], 1, 1, 1))
# Y_test = np.concatenate((loc_test, conf_test), axis=3)
# Y_test = transform(Y_test)

# def visualize_layer(model, layer_name):
#     """
#     Visualize the specified layer of the specified model.
#     """

#     # Get the layer.
#     layer = model.get_layer(layer_name)

#     # Get the kernels of the layer. Ignore the bias terms.
#     kernels = layer.get_weights()[0]
#     n_kernels = kernels.shape[3]

#     # Show the kernels as images.
#     for i in range(n_kernels):
#         plt.subplot(1, n_kernels, i + 1)
#         show_image(kernels[:, :, 0, i])

#     if AUGMENT:
#         plt.savefig("../img/augment/kernels.png")
#     else:
#         plt.savefig("../img/no_augment/kernels.png")

# https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py



# TODO: Average pooling?
# TODO: make the path input system better

content_path = "../examples/content_neckarfront.jpg"
style_path = "../examples/style_starrynight.jpg"


##############################
# MODEL ARCHITECTURE
##############################

# Load images.
content = preprocess_img(content_path)
style = preprocess_img(style_path)

# Declare variable to store the combined image.
width, height = load_img(content_path).size
combination = K.placeholder((1, height, width, 3))

# Concatenate the images into one tensor.
input_tensor = K.concatenate(content, style, combination)

# Load the pre-trained VGG19 network.
model = VGG19(input_tensor=input_tensor, include_top=False)


# # Calculate dimensions.
# input_width, input_height = content.size
# output_width, output_height = int(FIXED_HEIGHT * input_width / input_height), FIXED_HEIGHT


# base_output = base_model.output

# # Add new layers in place of the last layer in the original model.
# global1 = GlobalAveragePooling2D()(base_output)
# global1 = Reshape((1, 1, 2048))(global1)
# loc1 = Dense(4, activation='tanh')(global1)
# conf1 = Dense(1, activation='sigmoid')(global1)
# output1 = Concatenate(axis=3)([loc1, conf1])

# # Create the final model.
# model = Model(inputs=base_model.input, outputs=output1)


##############################
# TRAINING
##############################

# # Freeze original InceptionV3 layers during training.
# for layer in base_model.layers:
#     layer.trainable = False

# # Print summary and compile.
# model.summary()
# model.compile(loss=F, optimizer=OPTIMIZER)

# # Fit the model; save the training history and the best model.
# if SAVE:
#     checkpointer = ModelCheckpoint(filepath=RESULTS_DIR + "intermediate_model.hdf5", verbose=VERBOSE, save_best_only=True)
#     hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_data=(X_test, Y_test), callbacks=[checkpointer])
# else:
#     hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)

# model.save(RESULTS_DIR + "final_model.hdf5")
# np.save(RESULTS_DIR + "image_classification_results", hist.history)


##############################
# TESTING
##############################

# # Calculate test score and accuracy.
# score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

# print("_" * 65)
# print("Test loss: ", score)
# print("_" * 65)

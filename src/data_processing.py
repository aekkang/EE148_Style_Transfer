######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     data_processing.py
# Desc:     Defines functions for processing images to use as data
#           for style transfer.
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# from skimage.exposure import rescale_intensity

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import VGG19, preprocess_input
from keras import backend as K

from utility import *


##############################
# DATA PROCESSING
##############################

def preprocess_img(path):
    """
    Preprocess a single image at the given path into the VGG19
    input format.
    """

    # Load image.
    img = load_img(path)

    # Convert the image to a dataset.
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    # Use the VGG19 preprocess method.
    arr = preprocess_input(arr)

    # Contain the dataset in a Keras variable.
    var = K.variable(arr)

    return var

def deprocess_img(img):
    """
    Deprocess the given image to a consumable format.
    """

    # Reshape the image.
    width, height = img.size
    img = img.reshape((height, width, 3))

    # Undo the VGG19 preprocess method.
    img[:,:,0] += VGG_MEAN[0]
    img[:,:,1] += VGG_MEAN[1]
    img[:,:,2] += VGG_MEAN[2]

    # Clip the numbers to the appropriate range.
    img = np.clip(img, 0, 255).astype('uint8')

    # Swap channel order from RGB to BGR.
    img = img[:, :, ::-1]

    return img

# show_image(image * shifted / 255, show=False)
# plt.savefig("lol.png")
# cv2.imwrite("lol1.png", image * shifted / 255)

# def save_data():
#     """
#     Save the Caltech-UCSD Birds-200 dataset.
#     """

#     # Load relevant files.
#     image_paths = np.genfromtxt(DATA_DIR + "images.txt", dtype=None)
#     train_test_split = np.genfromtxt(DATA_DIR + "train_test_split.txt", dtype=None)
#     bounding_boxes = np.genfromtxt(DATA_DIR + "bounding_boxes.txt", dtype=None)

#     X_train, Y_train, X_test, Y_test = [], [], [], []

#     # Load and modify images.
#     for i, (image_id, image_path) in enumerate(image_paths):
#         if i % 100 == 0:
#             print("Saving image: " + str(i))

#         # Extract information.
#         train_test = train_test_split[i][1]

#         # Read and resize the image to the input size required by the network.
#         image_path = image_path.decode("UTF-8")
#         image = cv2.imread(IMAGE_DIR + image_path)
        
#         # Values used to recalculate scaled bounding box.
#         h, w, n_channels = image.shape
#         image, vpadding, hpadding, scale = resize_to_square(image)
#         hpadding /= scale
#         vpadding /= scale
#         side = float(max(w, h))

#         # Recalculate bonuding box.
#         box = bounding_boxes[i]
#         x, y = box[1] + math.floor(hpadding), box[2] + math.floor(vpadding)
#         dx, dy = box[3], box[4]
        
#         box = (x / side, y / side, (x + dx) / side, (y + dy) / side)

#         # Add the image and label to the datasets.
#         if train_test:
#             X_train.append(image)
#             Y_train.append(box)
#         else:
#             X_test.append(image)
#             Y_test.append(box)

#     np.save(PREPROCESSED_DIR + "X_train", X_train)
#     np.save(PREPROCESSED_DIR + "Y_train", Y_train)
#     np.save(PREPROCESSED_DIR + "X_test", X_test)
#     np.save(PREPROCESSED_DIR + "Y_test", Y_test)

# def export_data(boxes_train, boxes_test):
#     """
#     Exports the dataset from the output of the learned MultiBox.
#     """

#     (X_train, Y_train), (X_test, Y_test) = load_data()
    
#     # Crop and save.
#     X_train = [resize_to_square(crop_box(x, expand_box(boxes_train[i, :4], clip=True)))[0]
#                if boxes_train[i, 0] + 5 / 299. < boxes_train[i, 2]
#                   and boxes_train[i, 1] + 5 / 299. < boxes_train[i, 3]
#                else x
#                for i, x in enumerate(X_train)]
#     X_test = [resize_to_square(crop_box(x, expand_box(boxes_test[i, :4], clip=True)))[0]
#                if boxes_test[i, 0] + 5 / 299. < boxes_test[i, 2]
#                   and boxes_test[i, 1] + 5 / 299. < boxes_test[i, 3]
#                else x
#                for i, x in enumerate(X_test)]

#     X_train = np.array(X_train)
#     X_test = np.array(X_test)

#     return (X_train, Y_train), (X_test, Y_test)

# def load_data():
#     """
#     Load and return the Caltech-UCSD Birds-200 dataset.
#     """

#     X_train = np.load(PREPROCESSED_DIR + "X_train.npy")
#     Y_train = np.load(PREPROCESSED_DIR + "Y_train.npy")
#     X_test = np.load(PREPROCESSED_DIR + "X_test.npy")
#     Y_test = np.load(PREPROCESSED_DIR + "Y_test.npy")

#     return (X_train, Y_train), (X_test, Y_test)

# def load_classes():
#     """
#     Load and return the classes of the Caltech-UCSD Birds-200 dataset.
#     """

#     image_class_labels = np.genfromtxt(DATA_DIR + "image_class_labels.txt", dtype=None)[:, 1]
#     train_test_split = np.genfromtxt(DATA_DIR + "train_test_split.txt", dtype=None)[:, 1]
#     classes = np.genfromtxt(DATA_DIR + "classes.txt", dtype=None)

#     classes_dict = {}

#     # Create a dictionary mapping class IDs to their names.
#     for class_id, class_name in classes:
#         classes_dict[class_id - 1] = ' '.join(class_name.decode("UTF-8")[4:].split('_'))

#     # Get the classes of the train and test sets.
#     classes_train = image_class_labels[train_test_split == 1] - 1
#     classes_test = image_class_labels[train_test_split == 0] - 1

#     return classes_train, classes_test, classes_dict

# def resize_to_square(image):
#     """
#     Resize the given image to a 299x299 square, corresponding to an
#     input for InceptionV3.
#     """

#     # Determine new dimensions.
#     h, w, n_channels = image.shape
#     scale = INCEPTIONV3_SIZE / float(max(w, h))
#     new_dim = (int(scale * w), int(scale * h))
    
#     # Determine padding.
#     vpadding = (INCEPTIONV3_SIZE - new_dim[1]) / 2.
#     hpadding = (INCEPTIONV3_SIZE - new_dim[0]) / 2.

#     # Resize and pad image.
#     image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
#     image = cv2.copyMakeBorder(image,
#                                int(math.floor(vpadding)), int(math.ceil(vpadding)),
#                                int(math.floor(hpadding)), int(math.ceil(hpadding)),
#                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

#     return image, vpadding, hpadding, scale

# def crop_box(image, box):
#     """
#     Crop the given image to the given bounding box.
#     """

#     x1 = int(box[0])
#     y1 = int(box[1])
#     x2 = int(box[2])
#     y2 = int(box[3])

#     cropped = image[y1:y2, x1:x2]
    
#     return cropped


# if __name__ == "__main__":
#     # save_data()
#     (X_train, Y_train), (X_test, Y_test) = load_data()

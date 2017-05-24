######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     minimizer.py
# Desc:     Contains Minimizer class for style transfer.
######################################################################

import time
import numpy as np
from cv2 import imwrite

from data_processing import *
from config import *


##############################
# MINIMIZER CLASS
##############################

# At each iteration of the L-BFGS-B algorithm, the loss to minimize is
# computed at a point, and then the gradients of the loss is computed at
# the same point. Thus, a class is useful to have as an interface to
# store shared values and a callback method to use throughout the
# minimization process.

class Minimizer(object):
    """Interface to SciPy's minimize function.
    """

    def __init__(self, f_to_minimize, width, height, dir_path):
        """Initialize shared values and store the loss function to minimize.
        """

        # Shared values.
        self.loss = None
        self.gradients = None

        # Loss function to minimize.
        self.f_to_minimize = f_to_minimize
        
        # Values for the callback method.
        self.i = 0
        self.start = time.time()
        self.width = width
        self.height = height
        self.combination_prefix = dir_path + "combination"
        self.logfile = open(dir_path + "result.log", 'w')

        print('')

    def calculate_loss_and_gradients(self, combination_i):
        """Given an input array, compute the loss and its gradients.
        """

        # Calculate the loss and its gradients.
        combination_i = combination_i.reshape((1, self.height, self.width, 3))
        output = self.f_to_minimize([combination_i])

        # Store the loss and its gradients.
        self.loss = output[0]
        self.gradients = output[1].flatten().astype('float64')

    def f_loss(self, combination_i):
        """Given an input array, return the loss.
        """

        self.calculate_loss_and_gradients(combination_i)

        return self.loss

    def f_gradients(self, combination_i):
        """Given an input array, return the gradients of the loss.
        """

        return np.copy(self.gradients)

    def write(self, combination_i):
        """Callback method that saves the combined image every particular number of iterations.
        """

        # Log status.
        self.logfile.write("{}, {:g}\n".format(self.i, self.loss))

        # Save the combined image if a sufficient number of iterations has passed.
        if self.i % SAVE_PER_N_ITERS == 0:
            # Print status.
            print("Iteration: {}".format(self.i))
            print("Elapsed time: {:.2f}s".format(time.time() - self.start))
            print("Current loss: {:g}".format(self.loss))
            print('')

            # Save the combined image.
            img = np.copy(combination_i)
            imwrite(self.combination_prefix + "_{}.jpg".format(self.i), deprocess_img(img, self.width, self.height))
        
        # Check if the minimization has finished.
        if self.i == ITERS:
            # Print status.1
            print("Finished!")
            print('')
            
            # Save the final combined image.
            combination_final = combination_i.reshape((1, self.height, self.width, 3))
            imwrite(self.combination_prefix + "_final.jpg", deprocess_img(combination_final, self.width, self.height))

            # Close the log file.
            self.logfile.close()

        self.i += 1

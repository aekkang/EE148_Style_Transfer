######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     parameter_exploration.py
# Desc:     Contains script for style transfer with various
#           parameters.
######################################################################

import subprocess


##############################
# PARAMETERS
##############################

# Parameters to explore.
PARAMETERS_LST = [
    ["1e0", "1e2", "0"],
    ["1e0", "1e3", "0"],
    ["1e0", "1e4", "0"]
]

# Directory prefix.
DIR_PREFIX = "examples/neckarfront"


##############################
# PARAMETER EXPLORATION
##############################

# Call the style transfer script with different sets of parameters.
for parameters in PARAMETERS_LST:
    parameters.append(DIR_PREFIX)

    subprocess.call(["python", "src/style_transfer.py",
                     "--content_weight {}".format(parameters[0]),
                     "--style_weight {}".format(parameters[1]),
                     "--variation_weight {}".format(parameters[2]),
                     " {3}_{0}_{1}_{2}/".format(*parameters)])

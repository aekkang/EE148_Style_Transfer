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

for parameters in PARAMETERS_LST:
    subprocess.call("python src/style_transfer" \
                    + " --content_weight " + parameters[0] \
                    + " --style_weight " + parameters[1] \
                    + " --variation_weight " + parameters[2] \
                    + " {}_{}_{}_{}/".format(DIR_PREFIX, *parameters))

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
    # Tests.
    ["1e0", "1e2", "0", "1", "1", "1", "1", "1", "400"],

    # Actual runs.
    ["1e0", "1e2", "0"],
    ["1e0", "1e2", "0"],
    ["1e0", "1e2", "0"],
    ["1e0", "1e2", "0"],
    ["1e0", "1e2", "0"],
    ["1e0", "1e2", "0"],
    ["1e0", "1e3", "0"],
    ["1e0", "1e4", "0"]
]


##############################
# PARAMETER EXPLORATION
##############################

# Call the style transfer script with different sets of parameters.
for parameters in PARAMETERS_LST:
    subprocess.call(["python", "src/style_transfer.py",
                     "runs2/starryneckarfront",
                     "--content_weight", parameters[0],
                     "--style_weight", parameters[1],
                     "--variation_weight", parameters[2],
                     "--style_layers_weights", parameters[3], parameters[4], parameters[5], parameters[6], parameters[7],
                     "--height", parameters[8]])

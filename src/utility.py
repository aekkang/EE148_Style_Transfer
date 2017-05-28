######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     utility.py
# Desc:     Defines utility functions for style transfer.
######################################################################


##############################
# MISCELLANEOUS FUNCTIONS
##############################

def format_parameter(n):
    """Formats a numerical parameter to be in scientific notation format.
    """

    # Convert the parameter to scientific notation.
    n_str = "{:.0e}".format(n)

    # Convert the exponent to a plain integer format.
    i = n_str.index('e')
    power_str = str(int(n_str[i + 1:]))

    return n_str[:i + 1] + power_str

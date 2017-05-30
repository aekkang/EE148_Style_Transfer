######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     argument_processing.py
# Desc:     Contains argument processing function for style transfer.
######################################################################

import os
import glob

from defaults import *
from utility import *


def process_args(args):
    """Given the results of parse_args() of an ArgumentParse object, processes
    and tweaks the results.
    """

    ##############################
    # WEIGHT ARGUMENTS
    ##############################

    content_weight = args.content_weight if args.content_weight is not None else CONTENT_WEIGHT
    style_weight = args.style_weight if args.style_weight is not None else STYLE_WEIGHT
    variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT
    style_layers_weights = args.style_layers_weights if args.style_layers_weights is not None else STYLE_LAYERS_WEIGHTS
    style_layers_weights = [float(w) / sum(style_layers_weights) for w in style_layers_weights]


    ##############################
    # MISCELLANEOUS ARGUMENTS
    ##############################

    # Saving & loading arguments.
    load_previous = args.load_previous if args.load_previous is not None else LOAD_PREVIOUS
    save_per_n_iters = args.save_per_n_iters or SAVE_PER_N_ITERS

    # Miscellaneous arguments.
    height = args.height if args.height is not None else HEIGHT
    iters = args.iters if args.iters is not None else ITERS


    ##############################
    # DIRECTORY PATH ARGUMENTS
    ##############################

    # Get input directory path.
    input_dir = os.path.abspath(args.input_dir)

    # Construct output directory path.
    output_dir = input_dir
    output_dir += "_{}".format(format_parameter(content_weight))
    output_dir += "_{}".format(format_parameter(style_weight))
    output_dir += "_{}".format(format_parameter(variation_weight))

    output_dir += "_w"
    for w in style_layers_weights:
        output_dir += "{}_".format(format_parameter(w))

    output_dir += "h{:d}".format(height)

    # Make output directory.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    ##############################
    # INPUT PATH ARGUMENTS
    ##############################

    # Construct paths to content image.
    if load_previous:
        # Find the latest save file.
        saves = glob(os.path.join(output_dir, "combination_*"))
        saves_nums = [int(save.split('_')[-1][:-4]) for save in saves]
        latest_save = saves[np.argmax(saves_nums)]
        latest_save_num = max(saves_nums)
        content_path = latest_save
    else:
        latest_save_num = None
        content_path = os.path.join(input_dir, "content.jpg")

    # Construct paths to style images.
    style_paths = glob.glob(os.path.join(input_dir, "style*"))


    ##############################
    # RETURN ARGUMENTS
    ##############################

    return input_dir, content_weight, style_weight, variation_weight, \
           style_layers_weights, load_previous, save_per_n_iters, \
           height, iters, output_dir, latest_save_num, content_path, style_paths

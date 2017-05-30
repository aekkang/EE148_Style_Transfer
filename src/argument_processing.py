######################################################################
# EE 148 Project
#
# Author:   Andrew Kang
# File:     argument_processing.py
# Desc:     Contains argument processing function for style transfer.
######################################################################

import os
import glob

import numpy as np

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
    style_ratios = np.array(args.style_ratios) / np.sum(args.style_ratios) if args.style_ratios is not None else STYLE_RATIOS
    variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT
    style_layer_weights = np.array(args.style_layer_weights) / np.sum(args.style_layer_weights) if args.style_layer_weights is not None else STYLE_LAYER_WEIGHTS


    ##############################
    # MISCELLANEOUS ARGUMENTS
    ##############################

    # Saving & loading arguments.
    load_previous = args.load_previous if args.load_previous is not None else LOAD_PREVIOUS
    save_per_n_iters = args.save_per_n_iters if args.save_per_n_iters is not None else SAVE_PER_N_ITERS

    # Miscellaneous arguments.
    height = args.height if args.height is not None else HEIGHT
    iters = args.iters if args.iters is not None else ITERS


    ##############################
    # DIRECTORY PATH ARGUMENTS
    ##############################

    # Get input directory path.
    input_dir = os.path.abspath(args.input_dir)

    # Construct output directory path.
    output_dir = input_dir + '_'

    # Content weight suffix.
    output_dir += "c{}_".format(format_parameter(content_weight))

    # Style weight suffix.
    output_dir += "s{}_".format(format_parameter(style_weight))

    # Style ratios suffix.
    output_dir += "sr"
    for style_ratio in style_ratios:
        output_dir += "{}_".format(format_parameter(style_ratio))
    
    # Variation weight suffix.
    output_dir += "v{}_".format(format_parameter(variation_weight))

    # Style layers weights suffix.
    output_dir += "sl"
    for style_layer_weight in style_layer_weights:
        output_dir += "{}_".format(format_parameter(style_layer_weight))

    # Height suffix.
    output_dir += "h{:d}".format(height)


    ##############################
    # INPUT PATH ARGUMENTS
    ##############################

    # Construct paths to content image.
    if load_previous:
        # Find the latest save file.
        saves = glob.glob(os.path.join(output_dir, "combination_*"))
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
    # ARGUMENT VALIDATION
    ##############################

    if len(style_ratios) != len(style_paths):
        raise ValueError("Number of style weights must match number of style images.")
    if len(style_layer_weights) != len(STYLE_LAYERS):
        raise ValueError("Number of style layers weights must match number of style layers.")


    ##############################
    # RETURN ARGUMENTS
    ##############################

    # Make output directory.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    return content_weight, style_weight, style_ratios, variation_weight, style_layer_weights, \
           load_previous, save_per_n_iters, height, iters, latest_save_num, \
           input_dir, output_dir, content_path, style_paths

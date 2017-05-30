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
    style_weights = [float(style_weight) / sum(style_weights) for style_weight in args.style_weights] if args.style_weights is not None else STYLE_WEIGHTS
    variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT
    style_layer_weights = [float(style_layer_weight) / sum(style_layer_weights) for style_layer_weight in style_layer_weights] if args.style_layer_weights is not None else STYLE_LAYER_WEIGHTS


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
    output_dir = input_dir + '_'

    # Content weight suffix.
    output_dir += "c{}_".format(format_parameter(content_weight))

    # Style weights suffix.
    output_dir += "s"
    for style_weight in style_weights:
        output_dir += "{}_".format(format_parameter(style_weight))
    
    # Variation weight suffix.
    output_dir += "v{}_".format(format_parameter(variation_weight))

    # Style layers weights suffix.
    output_dir += "sl"
    for style_layers_weight in style_layer_weights:
        output_dir += "{}_".format(format_parameter(style_layer_weight))

    # Height suffix.
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
    n_styles = len(style_paths)


    ##############################
    # ARGUMENT VALIDATION
    ##############################

    if len(style_weights) != n_styles:
        raise ValueError("Number of style weights must match number of style images.")
    if len(style_layer_weights) != len(STYLE_LAYER_WEIGHTS):
        raise ValueError("Number of style layers weights must match number of style layers.")


    ##############################
    # RETURN ARGUMENTS
    ##############################

    return input_dir, content_weight, style_weights, variation_weight, style_layer_weights, \
           load_previous, save_per_n_iters, height, iters, output_dir, latest_save_num, \
           content_path, style_paths, n_styles

from utility import *


def process_args(args):
    # Path arguments.
    input_dir = os.path.abspath(args.input_dir)

    # Weight arguments.
    content_weight = args.content_weight if args.content_weight is not None else CONTENT_WEIGHT
    style_weight = args.style_weight if args.style_weight is not None else STYLE_WEIGHT
    variation_weight = args.variation_weight if args.variation_weight is not None else VARIATION_WEIGHT
    style_layers_weights = args.style_layers_weights if args.style_layers_weights is not None else STYLE_LAYERS_WEIGHTS

    if len(style_layers_weights) == len(STYLE_LAYERS):
        style_layers_weights = [float(w) / sum(style_layers_weights) for w in style_layers_weights]
    else:
        raise ValueError("Number of weights must match number of layers.")

    # Saving & loading arguments.
    load_previous = args.load_previous if args.load_previous is not None else LOAD_PREVIOUS
    save_per_n_iters = args.save_per_n_iters or SAVE_PER_N_ITERS

    # Miscellaneous arguments.
    height = args.height if args.height is not None else HEIGHT
    iters = args.iters if args.iters is not None else ITERS

    # Construct path to output directory.
    output_dir = os.path.join(input_dir, "../{}_c{}_s{}_v{}_w".format(input_dir.split('/')[-1],
                                format_parameter(content_weight),
                                format_parameter(style_weight),
                                format_parameter(variation_weight)))

    for w in style_layers_weights:
        output_dir += "{}_".format(format_parameter(w))

    output_dir += "h{:d}".format(height)

    output_dir = os.path.abspath(output_dir)

    # Make output directory.
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Construct paths to input files.
    if load_previous:
        # Find the latest save file.
        saves = glob(os.path.join(output_dir, "combination_*"))
        saves_nums = [int(previous.split('_')[-1]) for save in previous_saves]
        latest_save = previous_saves[np.argmax(save_nums)]
        latest_save_num = max(saves_nums)
        content_path = latest_save
    else:
        latest_save_num = None
        content_path = os.path.join(input_dir, "content.jpg")

    style_path = os.path.join(input_dir, "style.jpg")

    return input_dir, content_weight, style_weight, variation_weight,
           style_layers_weights, load_previous, save_per_n_iters,
           height, iters, output_dir, content_path, style_paths
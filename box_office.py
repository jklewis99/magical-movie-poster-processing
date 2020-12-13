parser = argparse.ArgumentParser(description=" Predict movie box office revenue.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    'mode',
    choices=['train', 'predict', 'find_threshold', 'class_activation_map'],
    help="This is a REQUIRED PARAMETER to set mode to 'train' or 'predict' or 'find_threshold' or 'class_activation_map'")
# mode - test & train(raise an error)
#get test data
# num of samples optional parameters

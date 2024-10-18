import  argparse


def get_arg_parser():
    # Receive paths from command line
    parser = argparse.ArgumentParser(description='Shared argument parser')
    parser.add_argument('path_dataframe_train', type=str, help='Path to the train dataframe')
    parser.add_argument('path_dataframe_test',  type=str, help='Path to the test dataframe')
    parser.add_argument('path_config_json',     type=str, help='Path to the config json file')
    args = parser.parse_args()

    return args


def main():
    args = get_arg_parser()
    return args


if __name__ == '__main__':
    main()

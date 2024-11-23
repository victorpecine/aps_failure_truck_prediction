import  argparse


def get_arg_parser():
    """
    The `get_arg_parser` function defines an argument parser for command line arguments related to file
    paths and MLflow settings.

    :return: The `get_arg_parser` function returns the parsed arguments from the command line using the
    `argparse` module in Python.
    """
    # Receive paths from command line
    parser = argparse.ArgumentParser(description='Shared argument parser')
    parser.add_argument('--path_dataframe_train',
                        type=str,
                        required=False,
                        help='Path to train dataframe'
                        )
    parser.add_argument('--path_dataframe_train_processed',
                        type=str,
                        required=False,
                        help='Path to processed train dataframe'
                        )
    parser.add_argument('--path_dataframe_test',
                        type=str,
                        required=False,
                        help='Path to test dataframe'
                        )
    parser.add_argument('--path_dataframe_test_processed',
                        type=str,
                        required=False,
                        help='Path to processed test dataframe'
                        )
    parser.add_argument('--path_dataframe_predict_proba',
                        type=str,
                        required=False,
                        help='Path to dataframe with probabilities predicted'
                        )
    parser.add_argument('--path_config_json',
                        type=str,
                        required=False,
                        help='Path to config JSON file',
                        default='src/config.json'
                        )
    parser.add_argument('--mlflow_set_tracking_uri',
                        type=str,
                        required=False,
                        help='URI of mlflow',
                        default='http://localhost:5000'
                        )
    parser.add_argument('--mlflow_experiment_name',
                        type=str,
                        required=False,
                        help='MLflow experiment name',
                        default=None
                        )
    parser.add_argument('--mlflow_model_name',
                        type=str,
                        required=False,
                        help='MLflow model name',
                        default=None
                        )

    args = parser.parse_args()

    return args


# def main():
#     args = get_arg_parser()
#     return args


# if __name__ == '__main__':
#     main()

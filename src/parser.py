import  argparse


def get_arg_parser():
    # Receive paths from command line
    parser = argparse.ArgumentParser(description='Shared argument parser')
    parser.add_argument('--path_dataframe_train',
                        type=str,
                        required=False,
                        help='Path to the train dataframe'
                        )
    parser.add_argument('--path_dataframe_validation',
                        type=str,
                        required=False,
                        help='Path to the validation dataframe',
                        default=None
                        )
    parser.add_argument('--path_dataframe_test',
                        type=str,
                        required=False,
                        help='Path to the test dataframe',
                        default=None
                        )
    parser.add_argument('--path_config_json',
                        type=str,
                        required=False,
                        help='Path to the config JSON file',
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
                        required=True,
                        help='MLFlow experiment name'
                        )
    parser.add_argument('--mlflow_model_name',
                        type=str,
                        required=True,
                        help='MLFlow model name'
                        )

    args = parser.parse_args()

    return args


def main():
    args = get_arg_parser()
    return args


if __name__ == '__main__':
    main()

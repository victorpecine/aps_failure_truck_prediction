import  mlflow
import  pandas as pd
from    sklearn.ensemble import RandomForestClassifier
from    parser import get_arg_parser
from    load_params import load_json


def train(dataframe_train, parameters):
    print('#' * 80)
    print('TRAIN STARTED')
    print('#' * 80)

    # Split X and y train
    target   = parameters['target']
    features = dataframe_train.drop(columns=target).columns
    X_train  = dataframe_train[features]
    y_train  = dataframe_train[target]

    # Create and fit model
    model = RandomForestClassifier(**parameters['model_parameters'])
    model.fit(X_train, y_train)

    # Log params metadata
    mlflow.log_params(parameters)

    # Convert list of features to dictionary
    features_to_log = {'features': features.values}
    mlflow.log_params(features_to_log)

    # Log model metadata
    signature = mlflow.models.infer_signature(dataframe_train[features])
    model_name = 'rand_forest'
    mlflow.sklearn.log_model(sk_model=model,
                                artifact_path=model_name,
                                signature=signature,
                                registered_model_name=model_name
                                )

    # Log dataframe metadata
    dataframe_train[target] = dataframe_train[target].astype(float)
    metadata_dataframe_train = mlflow.data.from_pandas(df=dataframe_train,
                                                       targets=target,
                                                       name='Train'
                                                       )
    mlflow.log_input(metadata_dataframe_train, context='Train')

    print('#' * 80)
    print('TRAIN COMPLETED')
    print('#' * 80)

    return None


# def main():
#     args = get_arg_parser()
#     # Load dataframes
#     df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',', na_values=['na'])
#     # Load params
#     params = load_json(args.path_config_json)

#     # Train model
#     train(df_train, params)


# if __name__ == '__main__':
#     main()

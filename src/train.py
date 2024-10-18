import  pandas as pd
import  mlflow
import  joblib
from    sklearn.ensemble import RandomForestClassifier
from    load_params import load_json
from    parser import get_arg_parser
from    preprocessing import scale_features
from    datetime import datetime


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('aps_truck_failure_prediction')


def train(dataframe_train, params):
    print('#' * 80)
    print('TRAIN STARTED')
    print('#' * 80)

    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Create and save scaler
    scaler = scale_features(dataframe_train, params)
    with open('scaler.bin', 'wb') as f:
        joblib.dump(scaler, 'scaler.bin')
    # Log the artifact
    mlflow.log_artifact('scaler.bin')

    target   = params['target']
    features = dataframe_train.drop(columns=target).columns
    X_train  = dataframe_train[features]
    X_train_scaled = scaler.transform(X_train)
    dataframe_train[target] = dataframe_train[target].astype(float)
    y_train_scaled = dataframe_train[target]

    # Create and fit model
    model = RandomForestClassifier(**params['model_parameters'])
    model.fit(X_train_scaled, y_train_scaled)

    # Initialize a mlflow run
    with mlflow.start_run(run_name=timestamp_id):
        # Log params metadata
        mlflow.log_params(params)

        # Convert list of features to dictionary
        features_to_log = {'features': features.values}
        mlflow.log_params(features_to_log)
        print('------------------METADATA PARAMS SAVED------------------')

        # Log model metadata
        signature = mlflow.models.infer_signature(dataframe_train[features])
        model_name = 'rand_forest'
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path=model_name,
                                 signature=signature,
                                 registered_model_name=model_name
                                 )
        print('------------------METADATA MODEL SAVED------------------')

        # Log dataframe metadata
        dataframe_train[target] = dataframe_train[target].astype(float)
        metadata_dataframe_train = mlflow.data.from_pandas(dataframe_train)
        mlflow.log_input(metadata_dataframe_train, context='train')
        print('------------------METADATA DATAFRAME SAVED------------------')

        print('#' * 80)
        print('TRAIN COMPLETED')
        print('#' * 80)

        return scaler


# def main():
#     args = get_arg_parser()

#     # Load dataframe
#     df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',')
#     # Load params
#     params = load_json(args.path_config_json)

#     # Train model
#     train(df_train, params)


# if __name__ == '__main__':
#     main()

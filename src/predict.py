import mlflow.tracking
import  pandas as pd
import  mlflow
import  os
import  shutil
from    load_params     import load_json
from    parser          import get_arg_parser
from    mlflow.tracking import MlflowClient
from    mlflow.models   import infer_signature


def predict_classification(dataframe_test, parameters, model_name: str, stage: str):
    """_summary_

    Args:
        dataframe_test (_type_): _description_
        parameters (_type_): _description_
        model_name (str): _description_
        stage (str): _description_

    Returns:
        _type_: _description_
    """
    print('#' * 80)
    print('PREDICT STARTED\n')

    # Load model from MLflow
    try:
        client = MlflowClient()
        # Get the model latest version of stage
        versions       = client.search_model_versions(f"name='{model_name}'")
        stage_versions = [v for v in versions if v.current_stage == stage]
        last_version   = stage_versions[0]
        print(f'>>>>>>>>> Using version {last_version.version} from stage {last_version.current_stage}\n')
        train_run_id         = last_version.run_id
        print(f'>>>>>>>>> Model run_id: {train_run_id}\n')
        model_uri = '/'.join(['models:', model_name, stage])
        model = mlflow.pyfunc.load_model(model_uri)

    except IndexError:
        # Create a version 1 model
        model_uri = '/'.join(['models:', model_name, stage])
        model = mlflow.pyfunc.load_model(model_uri)

    target   = parameters.get('target')
    features = dataframe_test.drop(columns=target).columns

    dataframe_test[target] = dataframe_test[target].astype(float)
    X_test = dataframe_test[features]
    y_test = dataframe_test[target]

    # Log parameters from used model
    client = mlflow.tracking.MlflowClient()
    train_params = client.get_run(train_run_id).data.params
    mlflow.log_params(train_params)

    # Create local directory to download artifacts
    artifacts_dir = os.path.join('train_artifacts', model_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    # Download artifacts from MLflow
    client.download_artifacts(run_id=train_run_id, path=model_name, dst_path=artifacts_dir)
    # upload_artifacts_path = os.path.join(artifacts_dir, model_name)
    mlflow.log_artifacts(artifacts_dir)

    # Predictions from test and get the event probabilities column
    y_prob_predict  = model.predict(X_test,
                                    {'predict_method': parameters.get('predict_method')}
                                    )[:, 1]
    df_prob_predict = pd.DataFrame(y_prob_predict, columns=['y_prob_predict'])
    df_prob_predict = df_prob_predict.join(y_test)
    df_prob_predict.rename(columns={'class': 'y_test'}, inplace=True)
    df_prob_predict = df_prob_predict.join(dataframe_test[features])

    # Log test dataframe metadata
    df_prob_predict['y_test']         = df_prob_predict['y_test'].astype(float)
    df_prob_predict['y_prob_predict'] = df_prob_predict['y_prob_predict'].astype(float)
    metadata_df_prob_predict          = mlflow.data.from_pandas(df=df_prob_predict,
                                                                targets='y_test',
                                                                name='Teste',
                                                                predictions='y_prob_predict'
                                                                )
    mlflow.log_input(metadata_df_prob_predict, context='Test')

    print('\nPREDICT COMPLETED')
    print('#' * 80)

    return df_prob_predict


# def main():
#     args = get_arg_parser()
#     # Load dataframe
#     df_test = pd.read_csv(args.path_dataframe_test, encoding='utf-8', sep=',')
#     # Load parameters
#     parameters = load_json(args.path_config_json)

#     model_name = 'rand_forest'  # Remove after use
#     stage      = 'Staging'  # Remove after use

#     # Predict from model
#     predict(df_test, parameters, model_name, stage)


# if __name__ == '__main__':
#     main()

import mlflow.tracking
import  pickle
import  pandas as pd
import  mlflow
import  os
from    load_params     import load_json
from    parser          import get_arg_parser
from    mlflow.tracking import MlflowClient


def predict_classification(dataframe_test, parameters, model_name: str, stage='None'):
    """
    The function `predict_classification` loads a model, makes predictions on a test dataset, logs
    parameters and metadata, and saves the predictions to a CSV file.
    
    :param dataframe_test: The `dataframe_test` parameter is a pandas DataFrame containing the test data
    for which you want to make predictions. It should include the features used for prediction as
    columns, as well as the target variable that you want to predict
    :param parameters: The `parameters` dictionary contains the following key-value pairs:
    :param model_name: The `model_name` parameter in the `predict_classification` function is used to
    specify the name of the model that you want to load and use for making predictions. This model
    should have been previously trained and saved using MLflow. The function will load the specified
    model based on the provided `model_name
    :type model_name: str
    :param stage: The `stage` parameter in the `predict_classification` function is used to specify the
    stage of the model to be loaded from MLflow. It is an optional parameter with a default value of
    `'None'`. This parameter allows you to specify a particular stage (e.g., 'Staging', ', defaults to
    None (optional)

    :return: The function `predict_classification` returns a pandas DataFrame `df_prob_predict`
    containing the predicted probabilities and actual target values for the test data, along with the
    features used for prediction.
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

    # Log parameters from model
    client = mlflow.tracking.MlflowClient()
    train_params = client.get_run(train_run_id).data.params
    mlflow.log_params(train_params)

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

    df_prob_predict_save_path = os.path.join('data', 'processed', 'df_prob_predict.csv')
    df_prob_predict.to_csv(df_prob_predict_save_path,
                           encoding='utf-8',
                           sep=',',
                           index=False
                           )

    return df_prob_predict


def main():
    args = get_arg_parser()
    # Load dataframe
    df_test = pd.read_csv(args.path_dataframe_test, encoding='utf-8', sep=',')
    # Load parameters
    parameters = load_json(args.path_config_json)

    model_name = 'rand_forest'  # Remove after use
    stage      = 'Staging'  # Remove after use

    # Predict from model
    predict_classification(df_test, parameters, model_name, stage)


if __name__ == '__main__':
    main()

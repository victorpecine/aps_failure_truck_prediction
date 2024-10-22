import  pandas as pd
import  mlflow
from    load_params import load_json
from    parser import get_arg_parser
from    mlflow.tracking import MlflowClient


def predict(dataframe_test, params, model_name: str, stage: str):
    print('#' * 80)
    print('PREDICT STARTED\n')

    # Load model from MLFlow
    try:
        client = MlflowClient()
        # Get the model latest version of stage
        versions       = client.search_model_versions(f"name='{model_name}'")
        stage_versions = [v for v in versions if v.current_stage == stage]
        last_version   = stage_versions[0]
        print(f'>>>>>>>>> Using version {last_version.version} from stage {last_version.current_stage}')
        model = mlflow.sklearn.load_model('/'.join(['models:', model_name, stage]))

    except IndexError:
        # Create a version 1 model
        model = mlflow.sklearn.load_model('/'.join(['models:', model_name, stage]))

    target   = params['target']
    features = dataframe_test.drop(columns=target).columns

    dataframe_test[target] = dataframe_test[target].astype(float)
    X_test = dataframe_test[features]
    y_test = dataframe_test[target]

    # Predictions from test
    # Select the event probabilities column
    y_prob_predict  = model.predict_proba(X_test)[:, 1]
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


def main():
    args = get_arg_parser()
    # Load dataframe
    df_test = pd.read_csv(args.path_dataframe_test, encoding='utf-8', sep=',')
    # Load params
    params = load_json(args.path_config_json)

    model_name = 'rand_forest'  # Remove after use
    stage      = 'Staging'  # Remove after use

    # Predict from model
    predict(df_test, params, model_name, stage)


if __name__ == '__main__':
    main()

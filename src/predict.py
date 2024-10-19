import  pandas as pd
import  mlflow
from    load_params import load_json
from    parser import get_arg_parser
from    mlflow.tracking import MlflowClient


client        = MlflowClient()
model_name    = 'rand_forest'
# Get the latest version of stage
stage         = 'None'
versions      = client.search_model_versions(f"name='{model_name}'")
none_versions = [v for v in versions if v.current_stage == stage]
last_version  = none_versions[0]
print(f'>>>>>>>>> Using version {last_version.version} from stage {last_version.current_stage}')


def predict(dataframe_test, params):
    print('#' * 80)
    print('PREDICT STARTED\n')

    # Load model
    last_version_model_path = '/'.join(['models:', model_name, str(last_version.version)])
    model    = mlflow.sklearn.load_model(last_version_model_path)

    target   = params['target']
    features = dataframe_test.drop(columns=target).columns
    dataframe_test[target] = dataframe_test[target].astype(float)

    X_test = dataframe_test[features]
    y_test = dataframe_test[target]

    y_predict  = model.predict(X_test)
    df_predict = pd.DataFrame(y_predict, columns=['y_predict'])
    df_predict = df_predict.join(y_test)
    df_predict.rename(columns={'class': 'y_test'}, inplace=True)

    df_test_predict = dataframe_test.join(df_predict['y_predict'])

    # Log dataframe metadata
    df_test_predict[target]       = df_test_predict[target].astype(float)
    df_test_predict['y_predict']  = df_test_predict['y_predict'].astype(float)
    metadata_df_test_predict      = mlflow.data.from_pandas(df=df_test_predict,
                                                            targets=target,
                                                            name='Train',
                                                            predictions='y_predict'
                                                            )
    mlflow.log_input(metadata_df_test_predict, context='Train')

    print('\nPREDICT COMPLETED')
    print('#' * 80)

    return df_predict


# def main():
#     args = get_arg_parser()
#     # Load dataframe
#     df_test = pd.read_csv(args.path_dataframe_test, encoding='utf-8', sep=',')
#     # Load params
#     params = load_json(args.path_config_json)

#     # Predict from model
#     predict(df_test, params)


# if __name__ == '__main__':
#     main()

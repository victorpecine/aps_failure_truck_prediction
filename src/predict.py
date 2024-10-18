import  pandas as pd
import  mlflow
import  json
import  joblib
import  requests
from    load_params import load_json
from    parser import get_arg_parser


mlflow.set_tracking_uri('http://localhost:5000')

def predict(dataframe_train, dataframe_test, params):
    print('#' * 80)
    print('PREDICT STARTED')
    print('#' * 80)

    # Load model
    model    = mlflow.sklearn.load_model('models:/rand_forest/staging')

    target   = params['target']
    features = dataframe_train.drop(columns=target).columns

    # Split test
    X_test   = dataframe_test[features]
    dataframe_test[target] = dataframe_test[target].astype(float)

    # Scale test features
    with open('scaler.bin', 'rb') as f:
        scaler = joblib.load(f)
    X_test_scaled  = scaler.transform(X_test)
    df_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    df_test_scaled = dataframe_test[[target]].join(df_test_scaled)
    # df_test_scaled_json = json.dumps(
    #     {'dataframe_records': df_test_scaled.to_dict(orient='records')}
    # )

    y_predict = model.predict(df_test_scaled[features])
    df_y_pred = pd.DataFrame(y_predict, columns=['y_predict'])
    df_y_pred = df_y_pred.join(dataframe_test[target])
    df_y_pred.rename(columns={'class': 'y_test'}, inplace=True)
    
    # # Request predictions
    # response = requests.post(url='http://127.0.0.1:5200/invocations',
    #                          data=df_test_scaled_json,
    #                          headers={'Content-Type': 'application/json'},
    #                          timeout=10
    #                          )

    # y_predict = pd.DataFrame(response.json(), index=df_test_scaled.index)
    # df_y_pred = y_predict.join(dataframe_test[target])
    # df_y_pred.rename(columns={'class': 'y_test', 'predictions': 'y_predict'}, inplace=True)
    print(df_y_pred)
    print('#' * 80)
    print('PREDICT COMPLETED')
    print('#' * 80)

    return df_y_pred


def main():
    args = get_arg_parser()

    # Load dataframes
    df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',')
    df_test = pd.read_csv(args.path_dataframe_test, encoding='utf-8', sep=',')
    # Load params
    params = load_json(args.path_config_json)

    # Predict from model
    predict(df_train, df_test, params)


if __name__ == '__main__':
    main()

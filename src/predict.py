import  pandas as pd
import  mlflow
from    load_params import load_json
from    parser import get_arg_parser


def predict(dataframe_test, params):
    print('#' * 80)
    print('PREDICT STARTED')
    print('#' * 80)

    # Load model
    model    = mlflow.sklearn.load_model('models:/rand_forest/staging')

    target   = params['target']
    features = dataframe_test.drop(columns=target).columns
    dataframe_test[target] = dataframe_test[target].astype(float)

    X_test = dataframe_test[features]
    y_test = dataframe_test[target]

    y_predict  = model.predict(X_test)
    df_predict = pd.DataFrame(y_predict, columns=['y_predict'])
    df_predict = df_predict.join(y_test)
    df_predict.rename(columns={'class': 'y_test'}, inplace=True)

    print('#' * 80)
    print('PREDICT COMPLETED')
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

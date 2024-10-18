import  pandas as pd
from    sklearn.preprocessing import StandardScaler
from    load_params import load_json
from    parser import get_arg_parser

def scale_features(dataframe_train, params):
    print('#' * 80)
    print('PREPROCESSING STARTED')
    print('#' * 80)

    target   = params['target']
    features = dataframe_train.drop(columns=target).columns
    X_train  = dataframe_train[features]

    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)

    print('#' * 80)
    print('PREPROCESSING COMPLETED')
    print('#' * 80)

    return scaler

# def main():
#     args = get_arg_parser()

#     # Load dataframe
#     df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',')
#     # Load params
#     params = load_json(args.path_config_json)

#     # Train model
#     scale_features(df_train, params)


# if __name__ == '__main__':
#     main()

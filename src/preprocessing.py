import  os
import  pandas as pd
import  numpy  as np
from    parser                 import get_arg_parser
from    load_params            import load_json
from    sklearn.impute         import SimpleImputer
from    imblearn.over_sampling import SMOTE
from    sklearn.preprocessing  import StandardScaler


RANDOM_SEED = np.random.seed(0)


def wrangling_data_columns(dataframe_train, parameters, dataframe_test=None):
    """
    Change target str to int as 0 (non event) and 1 (event)
    
    Use SimpleImputer to fill NaN values with train features medians
    
    Use SMOTE to balance train dataframe
    
    Drop features with zero variance from train and test
    
    Use StandardScaler to scale train and test
    
    Args:
        dataframe_train (pandas dataframe): Dataframe with features and target
        parameters (json): A file containing the parameters of model,
                            cross validation, target, train size and cutoff
        dataframe_test (pandas dataframe, optional): Dataframe with features and target. Defaults to None.

    Returns:
        df_train_scaled (pandas dataframe): Train dataframe scaled

        df_test_scaled (pandas dataframe): Test dataframe scaled
    """
    # Change strings for NaN
    dataframe_train.replace({'na': np.nan}, inplace=True)

    # Define target and features
    target   = parameters['target']
    features = dataframe_train.drop(columns=target).columns

    # Adjust features columns type
    dataframe_train[features] = dataframe_train[features].astype(float)

    # Change class to int dummies
    map_class = {'neg': 0, 'pos': 1}
    dataframe_train[target] = dataframe_train[target].map(map_class)

    # Wrangling data
    # Use the median to replace NaN values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(dataframe_train[features])

    # Fill train NaN with median imputer
    dataframe_train[features] = imputer.transform(dataframe_train[features])

    # Balance train data minority class
    X_train = dataframe_train[features]
    y_train = dataframe_train[target]
    oversample = SMOTE(sampling_strategy='minority', random_state=RANDOM_SEED)
    X_train_balanced, y_balanced = oversample.fit_resample(X_train, y_train)

    # Train dataframe with balanced data
    dataframe_train_balanced = X_train_balanced.join(y_balanced)

    # Remove zero variance train columns
    columns_to_drop = []
    X_train = dataframe_train_balanced.drop(columns=target)
    for i in X_train:
        if X_train[i].std() == 0:
            columns_to_drop.append(i)
    dataframe_train_zero_std = dataframe_train_balanced.drop(columns=columns_to_drop).copy()
    features_zero_std = dataframe_train_zero_std.drop(columns=target).columns

    # Scale data
    scaler = StandardScaler()
    scaler.fit(dataframe_train_zero_std[features_zero_std])

    # Data scaled
    X_train_scaled  = scaler.transform(dataframe_train_zero_std[features_zero_std])
    df_train_scaled = pd.DataFrame(X_train_scaled,
                                   columns=features_zero_std,
                                   index=dataframe_train_zero_std.index)
    df_train_scaled = dataframe_train_zero_std[[target]].join(df_train_scaled)

    # Check folder or create one to save train dataframe
    processed_data_path = os.path.join('data', 'processed_data')
    is_exist = os.path.exists(processed_data_path)
    if not is_exist:
        os.makedirs(processed_data_path)
        print(f'>>>>>>>>> Folder {processed_data_path} created successfully!')
    else:
        print(f'>>>>>>>>> Folder {processed_data_path} already exists.')

    df_train_scaled.to_csv(os.path.join(processed_data_path, 'df_train.csv'),
                           sep=',',
                           encoding='utf-8',
                           index=False
                           )

    # Wrangling on test dataframe if passed at args
    if not dataframe_test.empty:
        dataframe_test.replace({'na': np.nan}, inplace=True)
        dataframe_test[features] = dataframe_test[features].astype(float)
        dataframe_test[target]   = dataframe_test[target].map(map_class)
        # Fill test NaN with same train imputer
        dataframe_test[features] = imputer.transform(dataframe_test[features])
        # Remove the same column on test
        dataframe_test_zero_std = dataframe_test.drop(columns=columns_to_drop).copy()
        # Data scaled
        X_test_scaled  = scaler.transform(dataframe_test_zero_std[features_zero_std])
        df_test_scaled = pd.DataFrame(X_test_scaled,
                                      columns=features_zero_std,
                                      index=dataframe_test_zero_std.index)
        df_test_scaled = dataframe_test_zero_std[[target]].join(df_test_scaled)
        df_test_scaled.to_csv(os.path.join(processed_data_path, 'df_test.csv'),
                              sep=',',
                              encoding='utf-8',
                              index=False
                              )

        return df_train_scaled, df_test_scaled

    return df_train_scaled


# def main():
#     args = get_arg_parser()
#     # Load dataframes
#     df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',', na_values=['na'])
#     df_test  = pd.read_csv(args.path_dataframe_test,  encoding='utf-8', sep=',', na_values=['na'])
#     # Load params
#     params = load_json(args.path_config_json)
#     wrangling_data_columns(df_train, df_test, params)


# if __name__ == '__main__':
#     main()

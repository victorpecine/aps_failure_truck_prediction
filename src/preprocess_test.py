import  os
import  pickle
import  pandas as pd
import  numpy  as np
from    parser import get_arg_parser


def wrangling_test_data(path_dataframe_test: str):
    """
    Change target str to int as 0 (non event) and 1 (event)
    Use SimpleImputer to fill NaN values with test features medians
    Use SMOTE to balance test dataframe
    Drop features with zero variance from test and test
    Use StandardScaler to scale test and test
    
    Args:
        path_dataframe_test:
        parameters:
        path_dataframe_test:

    Returns:
        X_test_filled_nan (pandas dataframe): dataframe with test features only
        df_test (pandas dataframe): test dataframe processed
    """

    df_test = pd.read_csv(path_dataframe_test,
                          encoding='utf-8',
                          sep=',',
                          na_values=['na']
                         )

    train_features = pd.read_pickle('train_artifacts\\train_features.pkl')
    df_test_features = df_test[train_features].astype(float)
    df_test_target = df_test[['class']]
    # Change class to int dummies
    map_class = {'neg': 0, 'pos': 1}
    df_test_target.loc[:, 'class'] = df_test_target['class'].map(map_class)

    imputer_data_path = os.path.join('train_artifacts', 'median_imputer.pkl')
    with open(imputer_data_path, 'rb') as file:
        imputer = pickle.load(file)
        print('Imputer loaded successfully!')

    # Fill NaN on test
    X_test_filled_nan = imputer.transform(df_test_features)
    X_test_filled_nan = pd.DataFrame(X_test_filled_nan, columns=df_test_features.columns)

    # Save dataframes
    processed_data_path = os.path.join('data', 'processed')
    X_test_filled_nan.to_csv(os.path.join(processed_data_path, 'X_test_processed.csv'),
                             sep=',',
                             encoding='utf-8',
                             index=False
                            )
    print(f'>>>>>>>>> X_test_filled_nan saved on {processed_data_path}.')
    df_test_target.to_csv(os.path.join(processed_data_path, 'y_test_processed.csv'),
                          sep=',',
                          encoding='utf-8',
                          index=False
                          )
    print(f'>>>>>>>>> df_test_target saved on {processed_data_path}.')

    df_test_processed = df_test_target.join(X_test_filled_nan)
    df_test_processed.to_csv(os.path.join(processed_data_path, 'df_test_processed.csv'),
                             sep=',',
                             encoding='utf-8',
                             index=False
                            )
    print(f'>>>>>>>>> df_test_processed saved on {processed_data_path}.')

    return df_test_processed


def main():
    args = get_arg_parser()
    path_dataframe_test = args.path_dataframe_test
    wrangling_test_data(path_dataframe_test)


if __name__ == '__main__':
    main()

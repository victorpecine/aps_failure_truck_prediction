import  os
import  pickle
import  pandas as pd
import  numpy  as np
from    parser import get_arg_parser


def wrangling_test_data(path_dataframe_test: str):
    """
    The function `wrangling_test_data` reads test data, preprocesses it, saves the processed data, and
    returns the processed dataframe.
    
    :param path_dataframe_test: The function `wrangling_test_data` takes a path to a test dataframe as
    input. The test dataframe is read from the specified path, and certain preprocessing steps are
    applied to it. The processed data is then saved to CSV files in a specified directory
    :type path_dataframe_test: str

    :return: The function `wrangling_test_data` returns the processed test data stored in the DataFrame
    `df_test_processed`, which includes the target variable 'class' converted to integer dummies and the
    features with missing values filled using a median imputer.
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

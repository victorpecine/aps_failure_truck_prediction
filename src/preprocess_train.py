import  os
import  pickle
import  pandas as pd
import  numpy  as np
from    parser                 import get_arg_parser
from    sklearn.impute         import SimpleImputer
from    imblearn.over_sampling import SMOTE


RANDOM_SEED = np.random.seed(31)


def wrangling_train_data(path_dataframe_train: str):
    """
    Change target str to int as 0 (non event) and 1 (event)
    Use SimpleImputer to fill NaN values with train features medians
    Use SMOTE to balance train dataframe
    Drop features with zero variance from train and test
    Use StandardScaler to scale train and test
    
    Args:
        path_dataframe_train:
        parameters:
        path_dataframe_test:

    Returns:
        df_train (pandas dataframe): Train dataframe processed
    """

    # Load train dataframe
    df_train = pd.read_csv(path_dataframe_train,
                           encoding='utf-8',
                           sep=',',
                           na_values=['na'])

    # Create dataframe with percentage of NaN for each feature
    df_isna = df_train.drop(columns='class')
    df_isna = df_isna.isna().sum().sort_values(ascending=False).to_frame('total_nan')
    df_isna['pct_nan'] = (df_isna['total_nan'] / df_train.shape[0] * 100).round(2)
    # Remove features with over 50% of NaN data
    features_low_nan  = df_isna[df_isna['pct_nan'] <= 50].index.to_list()
    features_high_nan = df_isna[df_isna['pct_nan'] > 50].index.to_list()
    df_features = df_train[features_low_nan].astype(float)

    # Change class object to int dummies
    map_class = {'neg': 0, 'pos': 1}
    df_target = df_train['class'].map(map_class).to_frame('class').astype(int)

    # Identify features with zero variance
    features_zero_variance = [col for col in df_features if df_features[col].std() == 0]
    # Remove features with zero variance
    df_features.drop(columns=features_zero_variance, inplace=True)

    # Use the median to replace NaN values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(df_features)

    train_artifacts_data_path = os.path.join('train_artifacts')
    if not os.path.exists(train_artifacts_data_path):
        os.makedirs(train_artifacts_data_path)

    # Save the imputer to a file
    with open(os.path.join(train_artifacts_data_path, 'median_imputer.pkl'), 'wb') as file:
        pickle.dump(imputer, file)

    # Fill NaN on train
    X_train_filled_nan = imputer.transform(df_features)
    X_train_filled_nan = pd.DataFrame(X_train_filled_nan,
                                      columns=df_features.columns,
                                      index=df_features.index
                                      )

    # Balance train data minority class
    oversample = SMOTE(sampling_strategy='minority', random_state=RANDOM_SEED)
    oversample.fit(X_train_filled_nan, df_target)
    # Data balanced
    X_train_balanced, y_train_balanced = oversample.fit_resample(X_train_filled_nan, df_target)
    # Save train features
    train_features = X_train_balanced.columns
    with open(os.path.join(train_artifacts_data_path, 'train_features.pkl'), 'wb') as file:
        pickle.dump(train_features.to_list(), file)

    # y train form dataframe to series
    y_train_balanced = y_train_balanced['class']
    df_train_balanced = X_train_balanced.join(y_train_balanced)

    # Check folder or create one to save train dataframe
    processed_data_path = os.path.join('data', 'processed')
    is_exist = os.path.exists(processed_data_path)
    if not is_exist:
        os.makedirs(processed_data_path)

    df_train_balanced.to_csv(os.path.join(processed_data_path, 'df_train_processed.csv'),
                             sep=',',
                             encoding='utf-8',
                             index=False
                             )
    print(f'>>>>>>>>> df_train_balanced saved on {processed_data_path}.')
    import pdb; pdb.set_trace()
    return df_train_balanced


def main():
    args = get_arg_parser()
    path_dataframe_train = args.path_dataframe_train
    wrangling_train_data(path_dataframe_train)


if __name__ == '__main__':
    main()

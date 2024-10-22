import  mlflow
import  os
import  shutil
import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
from    sklearn.ensemble import RandomForestClassifier
from    sklearn.model_selection import RandomizedSearchCV
from    sklearn.model_selection import cross_validate


RANDOM_SEED = 0


def save_log_metadata(best_params, features, dataframe_train, model):
    # Log params metadata
    mlflow.log_param('Best parameters', best_params)

    # Convert list of features to dictionary
    features_to_log = {'features': features.values}
    mlflow.log_param('Features', features_to_log)

    # Log model metadata
    signature = mlflow.models.infer_signature(dataframe_train[features])
    model_name = 'rand_forest'
    mlflow.sklearn.log_model(sk_model=model,
                             artifact_path=model_name,
                             signature=signature,
                             registered_model_name=model_name
                             )
    return None


def train(dataframe_train, parameters):
    print('#' * 80)
    print('TRAIN STARTED\n')

    # Split X and y train
    target   = parameters['target']
    df_train = dataframe_train
    features = df_train.drop(columns=target).columns
    X_train  = df_train[features]
    y_train  = df_train[target]

    # Create and fit model
    model = RandomForestClassifier(random_state=RANDOM_SEED, verbose=0)

    # Random search on hyper parameters
    print('#' * 80)
    print('RANDOM SEARCH STARTED\n')
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=parameters['model_parameters'],
                                       verbose=3,
                                       cv=parameters['cross_validation']['folders'],
                                       n_iter=parameters['cross_validation']['n_iterations'],
                                       random_state=RANDOM_SEED,
                                       scoring='accuracy'
                                       )
    random_search.fit(X_train, y_train)
    # Define best parameters and apply on a new model
    best_params = random_search.best_params_
    print(f'\nBest parameters:\n{best_params}')
    model_best_params = RandomForestClassifier(**best_params, random_state=RANDOM_SEED)
    model_best_params.fit(X_train, y_train)

    print('\nRANDOM SEARCH COMPLETED')
    print('#' * 80)

    # Cross-validation
    print('#' * 80)
    print('CROSS VALIDATION STARTED\n')
    cv_results = cross_validate(estimator=model_best_params,
                                X=X_train,
                                y=y_train,
                                scoring=parameters['cross_validation']['scores'],
                                cv=parameters['cross_validation']['folders'],
                                verbose=3,
                                return_train_score=True,
                                error_score=np.nan
                                )

    # Save cross-validation scores
    for score in parameters['cross_validation']['scores']:
        cv_train_mean = cv_results[f'train_{score}'].mean().round(4)
        cv_test_mean  = cv_results[f'test_{score}'].mean().round(4)
        print(f'>>>>>>>>> CV {score} train mean: {cv_train_mean}')
        print(f'>>>>>>>>> CV {score} test mean: {cv_test_mean}')
        mlflow.log_metric(f'cv_{score}_train_mean', cv_train_mean)
        mlflow.log_metric(f'cv_{score}_test_mean',  cv_test_mean)

    print('\nCROSS VALIDATION COMPLETED')
    print('#' * 80)

    save_log_metadata(best_params,
                      features,
                      dataframe_train,
                      model_best_params)

    print('\nTRAIN COMPLETED')
    print('#' * 80)

    return None


# def main():
#     args = get_arg_parser()
#     # Load dataframes
#     df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',', na_values=['na'])
#     # Load params
#     params = load_json(args.path_config_json)

#     # Train model
#     train(df_train, params)


# if __name__ == '__main__':
#     main()

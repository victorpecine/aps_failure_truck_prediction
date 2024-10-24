import  mlflow
import  pandas as pd
import  numpy  as np
import  sys
from    sklearn.ensemble        import RandomForestClassifier
from    sklearn.model_selection import RandomizedSearchCV
from    sklearn.model_selection import cross_validate
# Append path to find the modules
sys.path.append('src\\')
from    parser          import get_arg_parser
from    load_params     import load_json
from    mlflow.models   import infer_signature
from    mlflow.pyfunc   import PythonModel


RANDOM_SEED = np.random.seed(0)


# Custom PythonModel class
class CustomRandomForestClassifier(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input, params=None):
        predict_method = params.get('predict_method')

        if predict_method == 'predict':
            return self.model.predict(model_input)
        elif predict_method == 'predict_proba':
            return self.model.predict_proba(model_input)
        elif predict_method == 'predict_log_proba':
            return self.model.predict_log_proba(model_input)
        else:
            raise ValueError(f'\n>>>>>>>>> The prediction method {predict_method} is not supported.')


def create_parameters_range(parameters: dict):
    range_n_estimators = np.arange(start=round(parameters.get('model_parameters')['n_estimators'][0], 2),
                                   stop=round(parameters.get('model_parameters')['n_estimators'][-1], 2),
                                   step=1
                                   )
    range_max_depth = np.arange(start=round(parameters.get('model_parameters')['max_depth'][0], 2),
                                stop=round(parameters.get('model_parameters')['max_depth'][-1], 2),
                                step=1
                                )
    range_min_samples_split = np.arange(start=round(parameters.get('model_parameters')['min_samples_split'][0], 2),
                                        stop=round(parameters.get('model_parameters')['min_samples_split'][-1], 2),
                                        step=0.1
                                        )
    range_min_samples_leaf = np.arange(start=round(parameters.get('model_parameters')['min_samples_leaf'][0], 2),
                                       stop=round(parameters.get('model_parameters')['min_samples_leaf'][-1], 2),
                                       step=1
                                       )
    param_distributions = {'n_estimators'     : range_n_estimators,
                           'max_depth'        : range_max_depth,
                           'min_samples_split': range_min_samples_split,
                           'min_samples_leaf' : range_min_samples_leaf
                          }

    return param_distributions


def train(dataframe_train: pd.DataFrame, parameters: dict):
    """
    _summary_

    Args:
        dataframe_train (_type_): _description_
        parameters (_type_): _description_

    Returns:
        _type_: _description_
    """
    print('#' * 80)
    print('TRAIN STARTED\n')

    # Split X and y train
    target   = parameters.get('target')
    df_train = dataframe_train
    dataframe_train[target] = dataframe_train[target].astype(float)
    features = df_train.drop(columns=target).columns
    X_train  = df_train[features]
    y_train  = df_train[target]

    # Create and fit model
    model = RandomForestClassifier(random_state=RANDOM_SEED, verbose=0)

    # Random search on hyper parameters
    param_distributions = create_parameters_range(parameters)
    print('#' * 80)
    print('RANDOM SEARCH STARTED\n')
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_distributions,
                                       verbose=3,
                                       cv=parameters.get('cross_validation')['folders'],
                                       n_iter=parameters.get('cross_validation')['n_iterations'],
                                       random_state=RANDOM_SEED,
                                       scoring='accuracy'
                                       )
    random_search.fit(X_train, y_train)
    # Define best parameters and apply on a new model
    best_params = random_search.best_params_
    print(f'\nBest parameters:\n{best_params}')

    model_best_params = RandomForestClassifier(**best_params, random_state=RANDOM_SEED)
    model_best_params.fit(X_train, y_train)

    # Define model signature
    signature = infer_signature(model_input=X_train[:2], params={'predict_method': 'predict_proba'})
    mlflow.pyfunc.log_model(artifact_path=parameters.get('model_name'),
                            python_model=CustomRandomForestClassifier(model_best_params),
                            signature=signature
                            )

    # Log best params metadata
    for param_name, value in best_params.items():
        mlflow.log_param(param_name, value)

    print('\nRANDOM SEARCH COMPLETED')
    print('#' * 80)

    # Cross-validation
    print('#' * 80)
    print('CROSS VALIDATION STARTED\n')
    cv_results = cross_validate(estimator=model_best_params,
                                X=X_train,
                                y=y_train,
                                scoring=parameters.get('cross_validation')['scores'],
                                cv=parameters.get('cross_validation')['folders'],
                                verbose=3,
                                return_train_score=True,
                                error_score=np.nan
                                )

    # Calculate cross-validation scores
    for score in parameters.get('cross_validation')['scores']:
        cv_train_mean = cv_results[f'train_{score}'].mean().round(4)
        cv_test_mean  = cv_results[f'test_{score}'].mean().round(4)
        print(f'>>>>>>>>> CV {score} train mean: {cv_train_mean}')
        print(f'>>>>>>>>> CV {score} test mean:  {cv_test_mean}')
        # Log cross-validation scores
        mlflow.log_metric(f'cv_{score}_train_mean', cv_train_mean)
        mlflow.log_metric(f'cv_{score}_test_mean',  cv_test_mean)

    print('\nCROSS VALIDATION COMPLETED')
    print('#' * 80)
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

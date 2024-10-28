import mlflow.artifacts
import  pandas as pd
import  numpy  as np
import  sys
import  os
import  mlflow
from    datetime      import datetime
# Append path to find the modules
sys.path.append('src\\')
from    load_params       import load_json
from    parser            import get_arg_parser
from    predict           import predict_classification
from    metrics           import calculate_metrics
from    metrics           import estimate_maintenance_costs
from    create_experiment import create_experiment
from    preprocessing     import wrangling_data


args = get_arg_parser()
# Arg paths
path_df_train   = args.path_dataframe_train
path_df_test    = args.path_dataframe_test
path_config     = args.path_config_json
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
model_name      = args.mlflow_model_name

# Load params
params = load_json(path_config)

# Access MLFLow
create_experiment(tracking_uri, experiment_name)

# Wrangling dataframe
_, df_test = wrangling_data(path_df_train, params, path_df_test)

# Execute the process for each cutoff
for cutoff in np.arange(0.2, 0.4, 0.01).round(2):
# Create run name with model name, cutoff and timestamp
    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f'test_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
    # Predict
    with mlflow.start_run(run_name=run_name):
        print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
        df_predict = predict_classification(df_test, params, model_name, 'Staging')
        true_negative, false_positive, \
            false_negative, true_positive = calculate_metrics(df_predict, cutoff)
        estimate_maintenance_costs(true_positive, false_negative, false_positive, params)

# Path and folder to save a file with predictions
predict_data_path = os.path.join('data', 'prediction')
is_exist = os.path.exists(predict_data_path)
if not is_exist:
    os.makedirs(predict_data_path)
    print(f'>>>>>>>>> Folder {predict_data_path} created successfully!')
else:
    print(f'>>>>>>>>> Folder {predict_data_path} already exists.')
# Save a pickle file with predictions
df_predict.to_pickle(os.path.join(predict_data_path, 'df_predict.pkl'))
print(f'>>>>>>>>> Dataframe with prediction saved at {predict_data_path}.')

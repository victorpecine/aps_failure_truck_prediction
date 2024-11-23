import  mlflow
import  sys
import  os
import  numpy as np
from    datetime      import datetime
# Add the src directory to sys.path
sys.path.append('src\\')
from    parser            import get_arg_parser
from    load_params       import load_json
from    create_experiment import create_experiment
from    preprocess_test   import wrangling_test_data
from    predict           import predict_classification
from    metrics           import calculate_metrics
from    metrics           import estimate_maintenance_costs


args = get_arg_parser()
# Load arg paths
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
path_df_test    = args.path_dataframe_test
path_config     = args.path_config_json
model_name      = args.mlflow_model_name

stage = 'Staging'

# Load params
params = load_json(path_config)

# Access to MLfLow
experiment_id = create_experiment(tracking_uri, experiment_name)

df_test_processed = wrangling_test_data(path_df_test)
df_prob_predict   = predict_classification(df_test_processed, params, model_name, stage)

# Path and folder to save a file with predictions
predict_data_path = os.path.join('data', 'prediction')
is_exist = os.path.exists(predict_data_path)
if not is_exist:
    os.makedirs(predict_data_path)
    print(f'>>>>>>>>> Folder {predict_data_path} created successfully!')
else:
    print(f'>>>>>>>>> Folder {predict_data_path} already exists.')
# Save a pickle file with predictions
df_prob_predict.to_pickle(os.path.join(predict_data_path, 'df_prob_predict.pkl'))
print(f'>>>>>>>>> Dataframe with prediction saved at {predict_data_path}.')

# If there's an active run, end it
if mlflow.active_run() is not None:
    mlflow.end_run()

# Calculate metrics for each cutoff
for cutoff in np.arange(0.05, 0.99, 0.05).round(2):
    # Create run name with model name, cutoff and timestamp
    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f'train_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
        tp, fn, fp = calculate_metrics(df_prob_predict, cutoff=cutoff)
        estimate_maintenance_costs(tp, fn, fp, params)

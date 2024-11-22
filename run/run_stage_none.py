import  mlflow
import  sys
from    datetime      import datetime
# Add the src directory to sys.path
sys.path.append('src\\')
from    parser            import get_arg_parser
from    load_params       import load_json
from    create_experiment import create_experiment
from    preprocess_train  import wrangling_train_data
from    preprocess_test   import wrangling_test_data
from    train             import train
from    predict           import predict_classification
from    metrics           import calculate_metrics
from    metrics           import estimate_maintenance_costs


# This code is used when it's necessary create a new model

args = get_arg_parser()
# Load arg paths
tracking_uri            = args.mlflow_set_tracking_uri
experiment_name         = args.mlflow_experiment_name
path_df_train           = args.path_dataframe_train
path_df_test            = args.path_dataframe_test
path_config             = args.path_config_json
model_name              = args.mlflow_model_name

stage = 'None'

# Load params
params = load_json(path_config)

# Access to MLfLow
experiment_id = create_experiment(tracking_uri, experiment_name)

df_train_processed = wrangling_train_data(path_df_train)
df_test_processed  = wrangling_test_data(path_df_test)

cutoff = 0.5
# Create run name with model name, cutoff and timestamp
timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f'train_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
    print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
    train(df_train_processed, params)
    df_prob_predict = predict_classification(df_test_processed, params, model_name, stage)
    tp, fn, fp = calculate_metrics(df_prob_predict)
    estimate_maintenance_costs(tp, fn, fp, params)

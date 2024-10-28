import  mlflow
import  sys
from    datetime      import datetime
# Add the src directory to sys.path
sys.path.append('src\\')
from    parser            import get_arg_parser
from    load_params       import load_json
from    train             import train
from    create_experiment import create_experiment
from    preprocessing     import wrangling_data


# This code is used when it's necessary create a new model

args = get_arg_parser()
# Load arg paths
path_df_train   = args.path_dataframe_train
config_path     = args.path_config_json
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
model_name      = args.mlflow_model_name

# Load params
params = load_json(config_path)

# Access to MLfLow
experiment_id = create_experiment(tracking_uri, experiment_name)

# Wrangling dataframe
df_train_scaled = wrangling_data(path_df_train, params)

cutoff = 0.5
# Create run name with model name, cutoff and timestamp
timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f'train_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
    print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
    train(df_train_scaled, params)

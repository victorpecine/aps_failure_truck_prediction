import  pandas as pd
import  mlflow
import  os
import  sys
from    datetime      import datetime
# Add the src directory to sys.path
sys.path.append('src\\')
from    preprocessing     import wrangling_data_columns
from    parser            import get_arg_parser
from    load_params       import load_json
from    train             import train
from    create_experiment import create_experiment


# This code is used when it's necessary create a new model
df_usage = 'train'
args = get_arg_parser()
# Load arg paths
path_df_train   = args.path_dataframe_train
path_df_test    = args.path_dataframe_test
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
model_name      = args.mlflow_model_name

# Load dataframes
df_train = pd.read_csv(path_df_train,
                       encoding='utf-8',
                       sep=','
                       )
df_test = pd.read_csv(path_df_test,
                      encoding='utf-8',
                      sep=','
                      )

# Load params
params = load_json(args.path_config_json)

# Access to MLFLow
experiment_id = create_experiment(tracking_uri, experiment_name)

# Wrangling dataframes
df_train_scaled, df_test_scaled = wrangling_data_columns(df_train, params, df_test)

cutoff = 0.5
# Create run name with model name, cutoff and timestamp
timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f'{df_usage}_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
    print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
    train(df_train_scaled, params)

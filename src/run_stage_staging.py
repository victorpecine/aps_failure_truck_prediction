import  pandas as pd
import  os
import  shutil
import  numpy as np
import  mlflow
from    preprocessing import wrangling_data_columns
from    parser import get_arg_parser
from    load_params import load_json
from    datetime import datetime
from    predict import predict
from    metrics import calculate_metrics
from    metrics import estimate_maintenance_costs


df_used = 'test'

args = get_arg_parser()
# Arg paths
path_df_train   = args.path_dataframe_train
path_df_test    = args.path_dataframe_test
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
model_name      = args.mlflow_model_name

stage           = 'Staging'

# Load dataframes
df_train = pd.read_csv(args.path_dataframe_train,
                       encoding='utf-8',
                       sep=','
                       )
df_test  = pd.read_csv(args.path_dataframe_test,
                       encoding='utf-8',
                       sep=','
                       )
# Load params
params = load_json(args.path_config_json)

# Access to MLFLow
try:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

except mlflow.exceptions.MlflowException:
    # Delete any file in mlflow trash folder
    path_mlflow_trash = os.path.join('mlruns', '.trash\\')
    if os.path.exists(path_mlflow_trash):
        shutil.rmtree(path_mlflow_trash)

    # Create a new experiment
    client = mlflow.MlflowClient()
    client.create_experiment(name=experiment_name)

# Wrangling dataframes
df_train_scaled, df_test_scaled = wrangling_data_columns(df_train, df_test, params)

# Execute the process for each cutoff
for cutoff in np.arange(0.05, 1, 0.05):
    # Create run name with model name, cutoff and timestamp
    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f'{df_used}_{model_name}_cutoff_{cutoff:.2f}_' + timestamp_id
    with mlflow.start_run(run_name=run_name):
        print(f'\n>>>>>>>>> APPLYING WITH CUTOFF {cutoff:.2f}')
        df_predict = predict(df_test_scaled, params, model_name, stage)
        true_negative, false_positive, \
            false_negative, true_positive = calculate_metrics(df_predict, cutoff)
        estimate_maintenance_costs(true_positive, false_negative, false_positive)

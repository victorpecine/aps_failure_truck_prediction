import  pandas as pd
import  mlflow
# import  matplotlib.pyplot as plt
from    preprocessing import wrangling_data_columns
from    parser import get_arg_parser
from    load_params import load_json
from    datetime import datetime
from    train import train
from    predict import predict
from    metrics import calculate_metrics


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_name='aps_truck_failure_prediction')

args = get_arg_parser()

# Load dataframes
df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',', na_values=['na'])
df_test  = pd.read_csv(args.path_dataframe_test,  encoding='utf-8', sep=',', na_values=['na'])
# Load params
params = load_json(args.path_config_json)

df_train_scaled, df_test_scaled = wrangling_data_columns(df_train, df_test, params)

timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
run_name = 'validation_'  + timestamp_id
with mlflow.start_run(run_name=run_name):
    train(df_train_scaled, params)
    df_predict = predict(df_test_scaled, params)
    calculate_metrics(df_predict)

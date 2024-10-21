import  pandas as pd
import  numpy as np
import  mlflow
from    preprocessing import wrangling_data_columns
from    parser import get_arg_parser
from    load_params import load_json
from    datetime import datetime
from    train import train
from    predict import predict
from    metrics import calculate_metrics
from    metrics import estimate_maintenance_costs


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_name='aps_truck_failure_prediction')

args = get_arg_parser()

# Load dataframes
df_train = pd.read_csv(args.path_dataframe_train,
                       encoding='utf-8',
                       sep=',',
                       na_values=['na'])
df_test  = pd.read_csv(args.path_dataframe_test,
                       encoding='utf-8',
                       sep=',',
                       na_values=['na'])
# Load params
params = load_json(args.path_config_json)

# Wrangling dataframes
df_train_scaled, df_test_scaled = wrangling_data_columns(df_train, df_test, params)

for cutoff in np.arange(0.05, 1, 0.05):
    # Create run name with model name, cutoff and timestamp
    timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    run_name = f'rand_forest_cutoff_{cutoff:.2f}_'  + timestamp_id
    # Execute the process for each cutoff
    with mlflow.start_run(run_name=run_name):
        print(f'\n>>>>>>>>> APPLYING WITH cutoff {cutoff:.2f}')
        train(df_train_scaled, params)
        df_predict = predict(df_test_scaled, params)
        true_negative, false_positive, \
            false_negative, true_positive = calculate_metrics(df_predict, cutoff)
        estimate_maintenance_costs(true_positive, false_negative, false_positive)

#TODO Add model to "staging" stage
#TODO Remove train() from cutoff loop
#TODO Apply dataframes pickle format (wrangling already done) on predict

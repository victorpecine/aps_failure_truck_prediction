import  pandas as pd
import  numpy  as np
import  json
import  requests
import  sys
import  os
from    sklearn.metrics import accuracy_score
from    sklearn.metrics import confusion_matrix
from    sklearn.metrics import precision_score
from    sklearn.metrics import recall_score
from    sklearn.metrics import f1_score
from    sklearn.metrics import roc_auc_score
# Append path to find the modules
sys.path.append('src\\')
from    parser            import get_arg_parser
from    load_params       import load_json
from    preprocess_test   import wrangling_test_data
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

# # Access to MLfLow
# experiment_id = create_experiment(tracking_uri, experiment_name)

df_test_processed = wrangling_test_data(path_df_test)

# Split X and y test
target   = params.get('target')
features = df_test_processed.drop(columns=target).columns

X_test = df_test_processed[features].astype(float)
y_test = df_test_processed[target].astype(int)

# Requisition format of MLFLow served model
X_test_json = json.dumps({'dataframe_records': X_test.to_dict(orient='records')})

# Model requisition via MLflow API do Mlflow and gest predictions as response
model_response = requests.post(url='http://127.0.0.1:5001/invocations',
                               data=X_test_json,
                               headers={'Content-Type': 'application/json'},
                               timeout=20
                               )

predictions     = model_response.json()
df_prob_predict = pd.DataFrame(predictions['predictions'],
                               columns=['_', 'y_prob_predict'])[['y_prob_predict']]
df_prob_predict = df_prob_predict.join(y_test)
df_prob_predict.rename(columns={'class': 'y_test'}, inplace=True)
# Create binary classification based on cutoff
cutoff = params.get('cutoff')

# Metrics
tp, fn, fp = calculate_metrics(df_prob_predict, cutoff=cutoff)
estimate_maintenance_costs(tp, fn, fp, params)

# Path and folder to save a file with probabilities predict
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

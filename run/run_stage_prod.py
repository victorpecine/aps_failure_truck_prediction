import  pandas as pd
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
from    create_experiment import create_experiment


df_used = 'test'
args = get_arg_parser()
# Arg paths
path_df_test    = args.path_dataframe_test
tracking_uri    = args.mlflow_set_tracking_uri
experiment_name = args.mlflow_experiment_name
model_name      = args.mlflow_model_name

# Load dataframes
df_test  = pd.read_csv(args.path_dataframe_test,
                       encoding='utf-8',
                       sep=','
                       )
# Load params
params = load_json(args.path_config_json)

# Access to MLFLow
create_experiment(tracking_uri, experiment_name)

# Split X and y test
target   = params.get('target')
features = df_test.drop(columns=target).columns

X_test = df_test[features]
y_test = df_test[target]

# Requisition format of MLFLow served model
X_test_json = json.dumps({'dataframe_records': X_test.to_dict(orient='records')})


# Model requisition via MLflow API do Mlflow and gest predictions as response
model_response = requests.post(url='http://127.0.0.1:5200/invocations',
                               data=X_test_json,
                               headers={'Content-Type': 'application/json'},
                               timeout=20
                               )

predictions = model_response.json()
df_predict  = pd.DataFrame(predictions['predictions'], columns=['y_predict'])
df_predict  = df_predict.join(df_test)
y_predict   = df_predict['y_predict']

cutoff = params.get('cutoff')

# Calculate metrics
accuracy       = accuracy_score(y_test, y_predict)
precision      = precision_score(y_test, y_predict, zero_division=0)
recall         = recall_score(y_test, y_predict, zero_division=0)
f1score        = f1_score(y_test, y_predict, zero_division=0)
auc_value      = roc_auc_score(y_test, y_predict)
tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
specificity    = tn / (tn + fp)

print(f'Cutoff:      {cutoff:.2f}')
print(f'Accuracy:    {accuracy:.2f}')
print(f'Precision:   {precision:.2f}')
print(f'Recall:      {recall:.2f}')
print(f'F1-score:    {f1score:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'AUC:         {auc_value:.2f}')
print(f'TN:          {tn:.2f}')
print(f'TP:          {tp:.2f}')
print(f'FN:          {fn:.2f}')
print(f'FP:          {fp:.2f}')

# Calculate costs
no_defect_cost  = params.get('maintenance_cost')['no_defect_cost']
preventive_cost = params.get('maintenance_cost')['preventive_cost']
corrective_cost = params.get('maintenance_cost')['corrective_cost']

no_defect_maintenance_cost  = fp * no_defect_cost
preventive_maintenance_cost = tp * preventive_cost
corrective_maintenance_cost = fn * corrective_cost

total_maintenance_cost = no_defect_maintenance_cost \
                            + preventive_maintenance_cost \
                                + corrective_maintenance_cost

print('\n')
print(f'Cost of no defect:              {no_defect_maintenance_cost:.2f}')
print(f'Cost of preventive maintenance: {preventive_maintenance_cost:.2f}')
print(f'Cost of corrective maintenance: {corrective_maintenance_cost:.2f}')
print(f'Total maintenance cost:         {total_maintenance_cost:.2f}\n')

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

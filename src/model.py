import  pandas as pd
import  mlflow
import  matplotlib.pyplot as plt
from    sklearn.ensemble import RandomForestClassifier
from    datetime import datetime
from    sklearn.metrics import accuracy_score
from    sklearn.metrics import confusion_matrix
from    sklearn.metrics import precision_score
from    sklearn.metrics import recall_score
from    sklearn.metrics import f1_score
from    sklearn.metrics import roc_auc_score
from    sklearn.metrics import ConfusionMatrixDisplay
from    sklearn.metrics import RocCurveDisplay
from    parser import get_arg_parser
from    load_params import load_json
from    sklearn.preprocessing import StandardScaler
from    mlflow.tracking import MlflowClient


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_name='Random forest - aps_truck_failure_prediction')

model_name = 'rand_forest'

args = get_arg_parser()

client = MlflowClient()

# Load params
params = load_json(args.path_config_json)

def get_zero_variance_features(dataframe, target_name: str):
    """
    Receive a dataframe and the target name
    Calculates the variance for each feature
    Return a list with the zero variance features
    """
    columns_to_drop = []
    X = dataframe.drop(columns=target_name)
    for i in X:
        if X[i].std() == 0:
            columns_to_drop.append(i)
            print(f'Column {i} has 0 variance and can be excluded')
    return columns_to_drop

def preprocessing(parameters):
    # Load dataframes
    df_train = pd.read_csv(args.path_dataframe_train, encoding='utf-8', sep=',')
    df_test  = pd.read_csv(args.path_dataframe_test,  encoding='utf-8', sep=',')

    # Remove features with zero variance
    train_columns_to_drop = get_zero_variance_features(df_train, 'class')
    df_train = df_train.drop(columns=train_columns_to_drop).copy()

    # Select same columns from train do test
    df_test = df_test[df_train.columns].copy()

    # Split train
    target_   = parameters['target']
    features_ = df_train.drop(columns=target_).columns
    X_train  = df_train[features_]
    df_train[target_] = df_train[target_].astype(float)

    # Split test
    X_test = df_test[features_]
    df_test[target_] = df_test[target_].astype(float)

    # Create scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Scale train
    X_train_scaled = scaler.transform(X_train)
    df_train_scaled_ = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    df_train_scaled_ = df_train_scaled_.join(df_train[target_])
    # Scale test
    X_test_scaled = scaler.transform(X_test)
    df_test_scaled_ = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    df_test_scaled_ = df_test_scaled_.join(df_test[target_])

    return df_train_scaled_, df_test_scaled_

df_train_scaled, df_test_scaled = preprocessing(params)

# Dataframe metadata
metadata_dataframe_train = mlflow.data.from_pandas(df_train_scaled)

target = params['target']
features = df_train_scaled.drop(columns=target).columns
features_to_log = {'features': features}
timestamp_id = datetime.now().strftime('%Y%m%d%H%M%S')
# Initialize a mlflow run
with mlflow.start_run(run_name='rand_forest'):
    # Log dataframe
    mlflow.log_input(metadata_dataframe_train, context='train')
    # Log params
    mlflow.log_params(params)
    # Log features
    mlflow.log_params(features_to_log)

    # Train
    print('>>>>>>> Train started')
    model = RandomForestClassifier(**params['model_parameters'])
    # import pdb; pdb.set_trace()

    model.fit(df_train_scaled[features], df_train_scaled[target])

    # Log model
    mlflow.sklearn.log_model(model, model_name)
    print('>>>>>>> Train completed')

    # Load model
    latest = client.get_latest_versions(model_name, stages=['Staging'])
    if latest:
        model_version = latest[0].version
        model_uri = latest[0].source
        model = mlflow.sklearn.load_model(model_uri)
        print(f'>>>>>>> Model version {model_version} loaded successfully')
    else:
        print('>>>>>>> No model found in the staging')

    # Predict
    print('>>>>>>> Predict started')
    y_predict = model.predict(df_test_scaled[features])
    print('>>>>>>> Predict completed')
    df_y_pred = pd.DataFrame(y_predict, columns=['y_predict'])
    df_y_pred = df_y_pred.join(df_test_scaled[target])
    df_y_pred.rename(columns={'class': 'y_test'}, inplace=True)
    # Log dataframe with predictions
    df_y_pred.to_pickle('data\\df_y_pred.pkl')
    mlflow.log_artifact('data\\df_y_pred.pkl')

    # Calculate metrics
    print('>>>>>>> Metric started')
    y_test         = df_y_pred['y_test']
    y_predict      = df_y_pred['y_predict']
    accuracy       = accuracy_score(y_test, y_predict)
    precision      = precision_score(y_test, y_predict, zero_division=0)
    recall         = recall_score(y_test, y_predict, zero_division=0)
    f1score        = f1_score(y_test, y_predict, zero_division=0)
    auc_value      = roc_auc_score(y_test, y_predict)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
    specificity    = tn / (tn + fp)

    plot_roc_curve = RocCurveDisplay.from_predictions(y_test, y_predict).plot()
    roc_fig, roc_ax = plt.subplots()
    plot_roc_curve.plot(ax=roc_ax)

    plot_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict)).plot()
    cm_fig, cm_ax = plt.subplots()
    plot_confusion_matrix.plot(ax=cm_ax)
    print('>>>>>>> Metric completed')

    # Log plots
    mlflow.log_figure(roc_fig, 'roc_curve.png')
    mlflow.log_figure(cm_fig,  'confusion_matrix.png')

    # Log metrics
    mlflow.log_metric('true_negative',  tn)
    mlflow.log_metric('true_positive',  tp)
    mlflow.log_metric('false_negative', fn)
    mlflow.log_metric('false_positive', fp)
    mlflow.log_metric('accuracy',       accuracy)
    mlflow.log_metric('precision',      precision)
    mlflow.log_metric('recall',         recall)
    mlflow.log_metric('f1score',        f1score)
    mlflow.log_metric('specificity',    specificity)
    mlflow.log_metric('auc',            auc_value)

import  pandas as pd
import  mlflow
import  matplotlib.pyplot as plt
from    predict import predict
from    sklearn.metrics import accuracy_score
from    sklearn.metrics import confusion_matrix
from    sklearn.metrics import precision_score
from    sklearn.metrics import recall_score
from    sklearn.metrics import f1_score
from    sklearn.metrics import roc_auc_score
from    sklearn.metrics import ConfusionMatrixDisplay
from    sklearn.metrics import RocCurveDisplay
from    load_params import load_json
from    parser import get_arg_parser


def calculate_metrics(dataframe_predict):
    print('#' * 80)
    print('METRICS STARTED\n')

    y_test    = dataframe_predict['y_test']
    y_predict = dataframe_predict['y_predict']

    # Calculate metrics
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

    # Save plots
    mlflow.log_figure(roc_fig, 'roc_curve.png')
    mlflow.log_figure(cm_fig,  'confusion_matrix.png')

    # Save metrics
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

    print('\nMETRICS COMPLETED')
    print('#' * 80)

    return None


# def main():
#     args = get_arg_parser()
#     # Metrics from model
#     calculate_metrics(df_train, df_test, params)


# if __name__ == '__main__':
#     main()

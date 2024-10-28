import  pandas as pd
import  numpy  as np
import  mlflow
import  matplotlib.pyplot as plt
from    sklearn.metrics import accuracy_score
from    sklearn.metrics import confusion_matrix
from    sklearn.metrics import precision_score
from    sklearn.metrics import recall_score
from    sklearn.metrics import f1_score
from    sklearn.metrics import roc_auc_score
from    sklearn.metrics import ConfusionMatrixDisplay
from    sklearn.metrics import RocCurveDisplay
from    load_params     import load_json
from    parser          import get_arg_parser


def calculate_metrics(dataframe_predict, cutoff=0.5):
    """
    _summary_

    Args:
        dataframe_predict (_type_): _description_
        cutoff (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    print('#' * 80)
    print('METRICS STARTED\n')

    # Define as an event the probabilities >= cutoff
    dataframe_predict['y_predict'] = np.where(dataframe_predict['y_prob_predict'] >= cutoff, 1, 0)
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

    # Plot curves
    # Area under curve
    plot_roc_curve = RocCurveDisplay.from_predictions(y_test, y_predict).plot()
    roc_fig, roc_ax = plt.subplots()
    plot_roc_curve.plot(ax=roc_ax)
    # Confusion matrix
    plot_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predict)).plot()
    cm_fig, cm_ax = plt.subplots()
    plot_confusion_matrix.plot(ax=cm_ax)
    # Confusion matrix with norm data based on true values
    plot_norm_true_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test,
                                                                              y_predict,
                                                                              normalize='true')
                                                                              )
    cmnt_fig, cmnt_ax = plt.subplots()
    plot_norm_true_confusion_matrix.plot(ax=cmnt_ax)
    # Confusion matrix with norm data based on predict values
    plot_norm_pred_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test,
                                                                              y_predict,
                                                                              normalize='pred')
                                                                              )
    cmnp_fig, cmnp_ax = plt.subplots()
    plot_norm_pred_confusion_matrix.plot(ax=cmnp_ax)
    # Confusion matrix with norm data based on all dataset
    plot_norm_all_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix(y_test,
                                                                             y_predict,
                                                                             normalize='all')
                                                                             )
    cmna_fig, cmna_ax = plt.subplots()
    plot_norm_all_confusion_matrix.plot(ax=cmna_ax)

    # Log cutoff
    mlflow.log_metric('cutoff',  cutoff)

    # Log plots
    mlflow.log_figure(cm_fig,   'confusion_matrix.png')
    mlflow.log_figure(cmnt_fig, 'confusion_matrix_true_normalized.png')
    mlflow.log_figure(cmnp_fig, 'confusion_matrix_pred_normalized.png')
    mlflow.log_figure(cmna_fig, 'confusion_matrix_all_normalized.png')

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

    return tn, fp, fn, tp


def estimate_maintenance_costs(true_positive: int, false_negative: int, false_positive: int, parameters: dict):
    """
    _summary_

    Args:
        true_positive (int): _description_
        false_negative (int): _description_
        false_positive (int): _description_

    Returns:
        _type_: _description_
    """
    no_defect_cost = parameters.get('maintenance_cost')['no_defect_cost']
    preventive_cost = parameters.get('maintenance_cost')['preventive_cost']
    corrective_cost = parameters.get('maintenance_cost')['corrective_cost']

    no_defect_maintenance_cost  = false_positive * no_defect_cost
    preventive_maintenance_cost = true_positive  * preventive_cost
    corrective_maintenance_cost = false_negative * corrective_cost

    total_maintenance_cost = no_defect_maintenance_cost \
                                + preventive_maintenance_cost \
                                    + corrective_maintenance_cost

    print('\n')
    print(f'Cost of no defect:              US$ {no_defect_maintenance_cost:.2f}')
    print(f'Cost of preventive maintenance: US$ {preventive_maintenance_cost:.2f}')
    print(f'Cost of corrective maintenance: US$ {corrective_maintenance_cost:.2f}')
    print(f'Total maintenance cost:         US$ {total_maintenance_cost:.2f}\n')
    print('#' * 80)

    # Log costs
    mlflow.log_metric('cost_no_defect_maintenance',  no_defect_maintenance_cost)
    mlflow.log_metric('cost_preventive_maintenance', preventive_maintenance_cost)
    mlflow.log_metric('cost_corrective_maintenance', corrective_maintenance_cost)
    mlflow.log_metric('cost_total_maintenance',      total_maintenance_cost)

    return None



# def main():
#     args = get_arg_parser()
#     # Metrics from model
#     calculate_metrics(df_train, df_test, params)


# if __name__ == '__main__':
#     main()

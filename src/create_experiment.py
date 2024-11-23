import  mlflow
import  os
import  shutil


def create_experiment(tracking_uri, experiment_name):
    """
    The function `create_experiment` sets up an experiment in MLflow with the specified tracking URI and
    name, handling exceptions and managing a `.trash` folder if needed.
    
    :param tracking_uri: The `tracking_uri` parameter in the `create_experiment` function is used to
    specify the URI where MLflow will store the experiment tracking data. This URI can point to a local
    directory, a remote server, or a cloud storage location where MLflow will log the experiment
    metrics, parameters, and artifacts
    :param experiment_name: The `experiment_name` parameter is a string that represents the name of the
    experiment you want to create or access in MLFlow. It is used to uniquely identify and organize
    different runs and results within MLFlow
    
    :return: The function `create_experiment` returns the experiment ID of the newly created experiment
    in MLFlow.
    """

    # Try to access the experiment at MLFLow
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name=experiment_name)

    except mlflow.exceptions.MlflowException:
        # Ensure the .trash folder exists
        path_mlflow_trash = os.path.join('mlruns', '.trash')
        # Clear files inside mlflow trash folder
        if os.path.exists(path_mlflow_trash):
            for filename in os.listdir(path_mlflow_trash):
                file_path = os.path.join(path_mlflow_trash, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        # Create trash folder if not exists
        if not os.path.exists(path_mlflow_trash):
            os.makedirs(path_mlflow_trash)
        # Create a new experiment
        client = mlflow.MlflowClient()
        experiment_id = client.create_experiment(name=experiment_name)

        return experiment_id

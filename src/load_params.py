import  os
import  json


def load_json(file_path):
    """
    The function `load_json` reads and loads a JSON file located in the same directory as the script.
    
    :param file_path: The `file_path` variable in the `load_json` function is initially passed as an
    argument to the function. However, within the function, the `file_path` variable is reassigned to
    the directory path of the current file using `os.path.dirname(__file__)`. 

    :return: The function `load_json` is returning the contents of the `config.json` file located in the
    same directory as the script where the function is called.
    """

    # Path to file.py
    file_path = os.path.dirname(__file__)
    params_path = os.path.join(file_path, 'config.json')
    with open(params_path, 'r', encoding='utf-8') as file:
        params = json.load(file)
    return params

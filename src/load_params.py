import  os
import  json


# Load JSON params from a file
def load_json(file_path):
    """
    _summary_

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Path to file.py
    file_path = os.path.dirname(__file__)
    params_path = os.path.join(file_path, 'config.json')
    with open(params_path, 'r', encoding='utf-8') as file:
        params = json.load(file)
    return params

import json
import mlflow

class Config:
    """
    Loads the settings file and set the values as attributes
    """
    def __init__(self):
        with open('settings.json', 'r') as f:
            settings = json.load(f)

            for key in settings:
                setattr(self, key, settings[key])

    def set_mlflow(self):
        """
        Store parameters in mlflow
        """
        with open('settings.json', 'r') as f:
            settings = json.load(f)

            for key in settings:
                if key != 'model':
                    mlflow.log_param(key, settings[key])
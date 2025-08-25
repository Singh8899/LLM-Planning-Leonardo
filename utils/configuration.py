from pathlib import Path
from .common_utils import load_yaml_file

def load_config():
    """
    Load configuration settings from a YAML file located in the 'config' directory.

    :return: Configuration settings as a dictionary.
    """
    config_path = Path(__file__).parent.parent.joinpath("config.yml")
    return load_yaml_file(config_path)


CONFIG = load_config()

import yaml


def load_yaml_file(file_path):
    """
    Load a YAML file and return its content.

    :param file_path: Path to the YAML file.
    :return: Content of the YAML file as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

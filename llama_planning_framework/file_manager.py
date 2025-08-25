import os
from pathlib import Path


class FileManager:
    """Manages file operations for PDDL files and generated plans.
    find_pddl_files: for each domain file, associates a list of problem file paths with it
    """

    def __init__(self):
        """Initialize the FileManager."""
        pass

    def read_file(self, file_path):
        """Read a file and return its contents.

        Args:
            file_path (str): Path to the file

        Returns:
            str: Contents of the file or None if error
        """
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def save_file(self, file_path, content):
        """Save content to a file.

        Args:
            file_path (str): Path to save the file
            content (str): Content to save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Saved plan to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")
            return False

    def find_pddl_files(self, problems_path):
        """Find all domain and problem files in the given directory.

        Args:
            problems_path (str): Path to search for PDDL files

        Returns:
            list: List of dictionaries containing domain and problem file information
        """
        pddl_directories = []
        for root, _, files in os.walk(problems_path):
            # Find domain files in current directory
            domain_files = [f for f in files if f.endswith("_domain.pddl")]
            if not domain_files:
                print(f"No domain files found in {root}")
                continue

            domain_path = os.path.join(
                root, domain_files[0]
            )  # usually only one domain file
            domain_text = self.read_file(domain_path)
            if domain_text is None:
                continue

            # Find problem files in current directory
            problem_paths = [
                os.path.join(root, f)
                for f in files
                if f.endswith(".pddl") and not f.endswith("_domain.pddl")
            ]  # all files other than domain
            if not problem_paths:
                print(f"No problem files found in {root}")
                continue

            domain_name = Path(domain_files[0]).stem.split("_domain")[0]

            pddl_directories.append(
                {  #
                    "domain_path": domain_path,
                    "domain_text": domain_text,
                    "domain_name": domain_name,
                    "problem_paths": problem_paths,  # list of all problem files for this domain
                }
            )

        return pddl_directories

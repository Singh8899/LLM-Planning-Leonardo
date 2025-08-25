import os
import sys
from pddl_processor import PDDLProcessor
from file_manager import FileManager
import argparse

# Add parent directory to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.configuration import CONFIG



class PDDLPlanner:
    """Main class for the PDDL Planning application."""

    def __init__(self, args):
        """Initialize the PDDLPlanner with parsed arguments."""
        self.args = args  # args passed from main
        self.model_manager = None  # model manager to load the model
        self.processor = None
        self.file_mng = FileManager()  # file manager to handle PDDL files

    def setup(self):
        """Set up the planner by loading the model and creating directories."""
        # Ensure output directory exists
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Initialize and load model
        if "Mistral" in self.args.weights_path:
            from model_manager_mistral import ModelManager
        else:
            from model_manager import ModelManager
        self.model_manager = ModelManager(
            self.args.weights_path
        )
        self.model_manager.load()

        # Initialize processor
        self.processor = PDDLProcessor(
            self.model_manager, self.args.output_dir
        )

    def run(self):
        """Run the PDDL planning process."""
        # Find PDDL files
        pddl_directories = self.file_mng.find_pddl_files(
            problems_path=self.args.problems_path
        )
        """pddl_directories:
            domain_path,
            domain_text,
            domain_name,
            problem_paths: list of problem paths. For each domain, it associates a list of problem file paths with it"""

        if not pddl_directories:
            print("No valid PDDL domains found. Exiting.")
            return

        # Process each domain
        for (
            domain_data
        ) in pddl_directories:  # iterate over the domains with associated problems
            self.processor.process_with_validation(domain_data, self.args)

        print("Processing complete!")


def main():
    """Main function to run the program."""
    parser = argparse.ArgumentParser(description="PDDL Planning with LLMs")

    parser.add_argument(
        "--problems_path",
        default=CONFIG["PROBLEMS_PATH"],
        help="Path to PDDL problem files",
    )
    parser.add_argument(
        "--weights_path",
        default=CONFIG["MODEL_PATH"],
        help="Path to model weights",
    )
    parser.add_argument(
        "--output_dir",
        default=CONFIG["MODEL_OUTPUT"],
        help="Directory to save outputs (defaults to ./model_outputs)",
    )
    parser.add_argument(
        "--batch",
        default=False,
        action="store_true",
        help="Process problems in batch mode",
    )
    parser.add_argument(
        "--include_prompt",
        default=True,
        action="store_true",
        help="Include the prompt in the output",
    )
    parser.add_argument(
        "--skip_special_tokens",
        default=False,
        action="store_true",
        help="Skip special tokens in the output",
    )
    parser.add_argument(
        "--cot",
        default=False,
        action="store_true",
        help="Use Chain of Thought (CoT) prompting",
    )
    parser.add_argument(
        "--sampling",
        default=False,
        action="store_true",
        help="Use sampling for generation (temperature 0.6, otherwise 0.0). Top_k: 10 for sampling",
    )
    parser.add_argument(
        "--add_system_prompt",
        action="store_true",
        default=True,
        help="Add system prompt to the input",
    )
    parser.add_argument(
        "--planner_validator",
        default=True,
        action="store_true",
        help="Use planner-validator framework. With planner-validator batch cannot be used.",
    )

    args = parser.parse_args()
    
    planner = PDDLPlanner(args)
    print("Setting up PDDL Planner with the following configuration:")
    planner.setup()
    print("PDDL Planner setup complete. Starting processing...")
    planner.run()


if __name__ == "__main__":
    print("Starting PDDL Planning Framework...")
    main()

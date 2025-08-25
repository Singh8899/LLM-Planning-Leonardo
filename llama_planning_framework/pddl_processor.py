import os
import prompts, file_manager
from pathlib import Path


class PDDLProcessor:
    """Processes PDDL problems and generates plans using language models.
    Uses prompts.py to create prompts for the model. Eg: ppdl_problem_prompt, cot_prompt
    """

    def __init__(self, model_manager, output_dir):
        """Initialize the PDDLProcessor.

        Args:
            model_manager (ModelManager): ModelManager instance
            output_dir (str): Directory to save outputs
        """
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.file_manager = file_manager.FileManager()

    def process_with_validation(self, domain_data, args):
        """Process each problem in a domain with validation of the plan.

        Args:
            domain_data (dict): Dictionary containing domain information
            args: Command line arguments
        """

        domain_text = domain_data["domain_text"]
        problem_paths = domain_data["problem_paths"
    ]  # list of all problem files for this domain
        domain_name = domain_data["domain_name"]

        # Create output directory for this domain
        domain_output_dir = os.path.join(self.output_dir, domain_name)
        os.makedirs(domain_output_dir, exist_ok=True)
        domain_data["domain_output_dir"] = domain_output_dir

        # Process each problem in problem_paths individually
        for problem_path in problem_paths:
            problem_text = self.file_manager.read_file(problem_path)
            if problem_text is None:
                continue

            print(f"Generating response for {problem_path}...")

            # uses the domain text and problem text to create the prompt
            problem_prompt = prompts.ppdl_problem_prompt(domain_text, problem_text)
            # if cot is true, it adds the chain of thought to the prompt
            if args.cot:
                problem_prompt = prompts.cot_prompt(problem_prompt)

            # Generate response
            print("Generating response...")
            # saving also the number of iterations for confidence evaluation
            response_text, iterations = self.model_manager.generate_with_validation(
                problem_prompt,
                problem_path,
                domain_data,
                max_tokens=5000,
                add_system_prompt=args.add_system_prompt,
                sampling=args.sampling,
                include_prompt=args.include_prompt,
                skip_special_tokens=args.skip_special_tokens,
            )
            response_text = (
                response_text + "\n\n" + f"Generated in {iterations} iterations."
            )
            # Save the plan
            problem_name = Path(problem_path).stem
            plan_path = os.path.join(domain_output_dir, f"{problem_name}_plan.txt")
            self.file_manager.save_file(plan_path, response_text)

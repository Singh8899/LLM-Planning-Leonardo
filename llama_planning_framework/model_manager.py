from multiprocessing.util import info
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import prompts
from pathlib import Path
import file_manager

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # getting parent directory to import validator
from utils.validator import validate_plan  # says wether plan is valid or not
from utils.answer_postprocessor import formatter  # extracts the answer

TRANSFORMERS_VERBOSITY=info

class ModelManager:
    """Manages loading and interaction with language models."""

    def __init__(self, weights_path):
        """Initialize the ModelManager with model name and path.

        Args:
            model_name (str): Name of the model to load ('llama' )
            weights_path (str): Path to the model weights
        """
        self.weights_path = weights_path
        self.model = None
        self.tokenizer = None
        self.weights_path = weights_path
        self.file_mng = file_manager.FileManager()

    def load(self):
        """Load the model and tokenizer.

        Returns:
            tuple: (model, tokenizer) loaded and ready to use
        """
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        print(
            f"CUDA {'is' if torch.cuda.is_available() else 'is not'} available. Using {device_info}."
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.weights_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",  # <-- fast attention
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.weights_path, padding_side="left"
            )
            return self.model, self.tokenizer
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            sys.exit(1)

    def generate_response(
        self,
        prompt,
        max_tokens=5000,
        add_system_prompt=True,
        sampling=False,
        include_prompt=True,
        skip_special_tokens=False,
    ):
        """Generate a response using the loaded model.

        Args:
            prompt (string): Prompt to process
            max_tokens (int): Maximum number of tokens to generate
            add_system_prompt (bool): Whether to add system prompt to the input
            sampling (bool): Whether to use sampling for generation
            include_prompt (bool): Whether to include the prompt in the output
            skip_special_tokens (bool): Whether to skip special tokens in the output

        Returns:
            str: Generated response text
        """
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model and tokenizer must be loaded before generating responses"
            )

        # adds the system prompt to the input if in args
        if add_system_prompt:
            messages = [
                {"role": "system", "content": prompts.system_prompt_pddl},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        formatted_message = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted_message, return_tensors="pt", truncation=False
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=sampling,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            cache_implementation="dynamic",
            use_cache=True,
        )

        response_tokens = (
            outputs[0][inputs["input_ids"].shape[1] :]
            if not include_prompt
            else outputs[0]
        )
        return self.tokenizer.decode(
            response_tokens, skip_special_tokens=skip_special_tokens
        )

    # for bathch generation changes message formatting and results output
    def batch_generate(
        self,
        prompts_list,
        max_tokens=5000,
        add_system_prompt=True,
        sampling=False,
        include_prompt=True,
        skip_special_tokens=False,
    ):
        """Generate responses for multiple prompts in batch.

        Args:
            prompts_list (list): List of prompts to process
            max_tokens (int): Maximum number of tokens to generate
            add_system_prompt (bool): Whether to add system prompt to the input
            sampling (bool): Whether to use sampling for generation
            include_prompt (bool): Whether to include the prompt in the output
            skip_special_tokens (bool): Whether to skip special tokens in the output

        Returns:
            list: List of generated response texts
        """
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model and tokenizer must be loaded before generating responses"
            )

        if add_system_prompt:
            formatted_messages = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompts.system_prompt_pddl},
                        {"role": "user", "content": problem},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in prompts_list
            ]
        else:
            formatted_messages = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for problem in prompts_list
            ]

        inputs = self.tokenizer(
            formatted_messages, return_tensors="pt", padding=True, truncation=False
        ).to(self.model.device)

        temperature = 0.6 if sampling else 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=sampling,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        results = []  # List to store generated texts
        for i, out in enumerate(outputs):
            gen = out[inputs["input_ids"].shape[1] :] if not include_prompt else out
            text = self.tokenizer.decode(gen, skip_special_tokens=skip_special_tokens)
            results.append(text)

        return results

    # uses generate_response, but adds validation logic
    def generate_with_validation(
        self,
        prompt,
        problem_path,
        domain_data,
        max_tokens=5000,
        add_system_prompt=True,
        sampling=False,
        include_prompt=True,
        skip_special_tokens=False,
    ):
        """Generate a response and validate it.

        Args:
            prompt (string): Prompt to process
            max_tokens (int): Maximum number of tokens to generate
            add_system_prompt (bool): Whether to add system prompt to the input
            sampling (bool): Whether to use sampling for generation
            include_prompt (bool): Whether to include the prompt in the output
            skip_special_tokens (bool): Whether to skip special tokens in the output

        Returns:
            str: Generated response text
        """
        MAX_ITERATIONS = 4  # maximum number of iterations to generate a valid plan
        iterations = 0

        # Initialize conversation history
        messages = []
        if add_system_prompt:
            messages.append({"role": "system", "content": prompts.system_prompt_pddl})
        messages.append({"role": "user", "content": prompt})

        past_key_values = None  # Store past keys for efficiency

        for i in range(MAX_ITERATIONS):
            iterations += 1
            print(f"Generating response for iteration {iterations}...")

            # Format messages for current iteration
            formatted_message = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                formatted_message, return_tensors="pt", truncation=False
            ).to(self.model.device)

            # Generate response with past_key_values for efficiency
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=sampling,
                temperature=0.6 if sampling else 0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                cache_implementation="dynamic",
                use_cache=True,
                past_key_values=past_key_values if i > 0 else None,
            )

            # Extract response tokens and decode
            response_tokens = outputs[0][
                inputs["input_ids"].shape[1] :
            ]  # skipping the prompt
            response = self.tokenizer.decode(
                response_tokens, skip_special_tokens=True
            )  # skip_special_tokens=skip_special_tokens)

            # Update past_key_values for next iteration
            if hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values

            # plan = formatter(response)  # extracting the plan from the response, skipping the prompt and special tokens

            # processing in order to have the correct paths to pass to the validator_____________
            domain_path = domain_data["domain_path"]
            domain_output_dir = domain_data["domain_output_dir"]
            problem_name = Path(problem_path).stem
            temp_plan_path = os.path.join(
                domain_output_dir, f"{problem_name}_temp{i}.txt"
            )
            self.file_mng.save_file(temp_plan_path, response)

            # validator takes in input the paths
            is_valid, reason = validate_plan(domain_path, problem_path, temp_plan_path)

            # breaks the loop if the plan is valid
            if is_valid:
                break

            print(f"Plan is not valid. Reason: {reason}. Retrying...")

            # Add assistant extracted response (plan) to conversation history
            messages.append({"role": "assistant", "content": response})

            # extracting reason from the validator
            if "No matching action" in reason:
                reason = (
                    "No matching action "
                    + reason.split("No matching action")[1].strip()
                )
                reason = reason.split("Errors")[0].strip()
            elif "Plan failed" in reason:
                reason = reason.split("Plan failed")[1].strip()
                reason = "Plan failed " + reason
                reason = reason.split("Failed plans")[0].strip()
            elif "Bad plan" in reason:
                reason = reason.split("Bad plan")[1].strip()
                reason = "Bad plan " + reason
                reason = reason.split("Failed plans")[0].strip()
            elif "Plan invalid" in reason:
                reason = reason.split("Plan invalid")[1].strip()
                reason = "Plan invalid " + reason
                reason = reason.split("Failed plans")[0].strip()
            else:
                reason = "Plan invalid"

            # Add validator feedback as user message for next iteration
            feedback_message = f"The plan you provided is invalid. {reason}. Please provide a corrected plan."
            messages.append({"role": "user", "content": feedback_message})

        return response, iterations

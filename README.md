## Environment and Requirements

- **Python version:** 3.11 (recommended). The code has been developed and tested with Python 3.11+, but you should align this with your cluster's available interpreter.
- **Dependencies:** All required Python packages are listed in `requirements.txt` (use the pinned versions there for reproducibility).

To set up a local virtual environment and install dependencies:

```bash
python3.11 -m venv _env
source _env/bin/activate
pip install -r requirements.txt
```

Notes:
- Some heavy ML packages (PyTorch + CUDA toolkits) are required for model execution on GPU nodes; follow your HPC or cluster instructions to install GPU-compatible builds.
- Optional plotting: `matplotlib` is used by analysis scripts; install it if you want figures generated.

---

# Llama Planning Framework on LEONARDO HPC (CINECA)

This repository provides a framework for running large language model (LLM) inference (Llama 3.3 70B) on planning tasks, with automated plan validation using the VAL validator. It is designed for use on the LEONARDO HPC system at CINECA Bologna, with scripts for node allocation and job execution.

## Features

- **LLM Planning**: Generates plans for PDDL planning tasks using Llama 3.3 70B models.
- **Plan Validation**: Uses the VAL validator to check the correctness of generated plans.
- **Batch Processing**: Supports batch and individual processing of planning problems.
- **HPC Integration**: Includes scripts for node allocation and job execution on LEONARDO HPC.

## Project Structure

- `llama_planning_framework/`
  - `main.py`: Entry point for running planning and validation.
  - `model_manager.py`: Loads and manages Llama models.
  - `model_manager_mistral.py`: Loads and manages Mistral models.
  - `pddl_processor.py`: Handles PDDL parsing and prompt generation.
  - `file_manager.py`: File I/O utilities.
  - `prompts.py`: Prompt templates for LLMs.
- `utils/`
  - `validator.py`: Python wrapper for the VAL validator.
  - `answer_postprocessor.py`: Extracts and formats LLM outputs.
  - `common_utils.py`: YAML config loader.
  - `configuration.py`: Configuration management utilities.
  - `dataset_generator.py`: Batch dataset generation and validation utilities. Also includes analysis utilities that justify the choice of 4 iterations (produces a text report, CSV table and optional plots).
- `scripts/`
  - `allocBoost.sh`: Example SLURM allocation command for LEONARDO.
- `meta_model/`: Meta-learning models and notebooks for plan success prediction.
- `report/`: Documentation and analysis reports.
- Root configuration files:
  - `config.yml`: Path to VAL binary and other configuration.
  - `requirements.txt`: Python dependencies.
  - `LICENSE`: Project license.
  - `TO INSTALL AND COMPILE VAL.txt`: VAL validator installation instructions.
  - `TO _INSTALL_NEW_ WEIGHTS.txt`: Model weights installation guide.
- `generated_plans/`, `problem_dataset/`, `generated_datasets_metamodel/`: Data and results folders.

## Running on LEONARDO HPC

1. **Allocate a Node**  
   Use the provided SLURM script to allocate a node with GPUs:
   ```bash
   salloc --partition=boost_usr_prod --qos=normal --nodes=1 --cpus-per-task=8 --gres=gpu:4 --time=16:00:00 --account=IscrC_ArtLLMs
   ```

2. **Activate Environment and Run**  
   Example workflow:
   ```bash
   cd /path/to/your/workspace/
   source _env/bin/activate
   python llama_planning_framework/main.py --model llama --problems_path <path_to_problems> --weights_path <path_to_weights> --output_dir <output_dir>
   ```

3. **Configuration**  
   - Edit `config.yml` to set the path to the VAL validator and other settings.
   - Model weights should be available at the specified `--weights_path`.

## Dataset generation & 4-iteration analysis

The repository includes `utils/dataset_generator.py` which:

- Collects generated plans from `generated_plans/` and validates them using VAL.
- Writes a consolidated CSV dataset to `generated_datasets_metamodel/` (e.g. `blocksworld_generations_dataset_llama.csv`).
- Produces a short analysis report and a CSV table that justify the empirical choice of using 4 iterations when generating plans; the analysis artifacts are saved under `analysis_results/` by default:
  - `iteration_analysis_report.txt` — human-readable justification and summary
  - `marginal_error_reduction_table.csv` — tabular marginal gains and error reductions per iteration
  - `iteration_justification_plots.png` — optional visualization (requires `matplotlib`)

Quick run (from repository root):

```bash
source _env/bin/activate
python utils/dataset_generator.py
```

Notes and assumptions made by `dataset_generator.py`:
- It expects plan files named like `<problem>_temp0.txt`, `<problem>_temp1.txt`, ... `<problem>_temp3.txt` (4 attempts) under the relevant `generated_plans/...` subfolders.
- It expects problem PDDL files under `problem_dataset/...` matching folder names.
- The script will skip problems or folders that are missing required files and will record `None` in generated fields.
- The analysis criteria used for success/marginal calculations are simplistic (e.g., 0 logical violations and >= 90% valid action percentage), and can be tuned inside the script.

If you want the plots but don't have `matplotlib` installed, install it in your environment:

```bash
pip install matplotlib
```

## Command-Line Arguments

- `--model`: Model to use (only `llama` supported for now)
- `--problems_path`: Path to PDDL problem files
- `--weights_path`: Path to model weights
- `--output_dir`: Directory to save outputs
- `--batch`: Enable batch processing
- `--planner_validator`: Enable plan validation

## Plan Validation

The framework uses the VAL validator (see `config.yml`) to check the validity of generated plans. Validation results are included in the output.

## Common troubleshooting / quick checks

- If the script cannot find VAL, make sure `config.yml` `VAL_PATH` points to the validator binary.
- If plan files are not picked up, verify `generated_plans/` subfolder layout and filename patterns.
- If running on GPU nodes, ensure suitable PyTorch + CUDA versions are installed (see `requirements.txt` for pinned packages used during development).


## References

- [VAL Validator](https://github.com/KCL-Planning/VAL)
- [Llama 3.3 70B Instruct](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [Mistral-Small-3.2-24B-Instruct-25063](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Developed as part of a student project at the University of Bologna, 2025.
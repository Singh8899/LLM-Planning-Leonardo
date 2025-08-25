import os
import subprocess
import argparse
from utils.configuration import CONFIG


def validate_plan(domain_file, problem_file, plan_file):

    result = subprocess.run(
        [CONFIG["VAL_PATH"], domain_file, problem_file, plan_file],
        capture_output=True,
        text=True,
    )
    if "Plan valid" in result.stdout:
        return True, None
    else:
        return False, result.stdout.strip()


def process_files(problem_dir: str, plan_dir: str) -> None:
    cnt = 0
    correct = 0
    for root, dirs, files in os.walk(plan_dir):
        for file in files:
            if file.endswith(".txt"):
                plan_path = os.path.join(root, file)

                base_name = file.split("_plan.txt")[0]

                relative_path = os.path.relpath(root, plan_dir)
                problem_path = os.path.join(
                    problem_dir, relative_path, base_name + ".pddl"
                )
                if not os.path.exists(problem_path):
                    print(
                        f"No corresponding .pddl file found for plan: {relative_path}/{base_name}"
                    )
                    continue
                domain_dir = os.path.join(problem_dir, relative_path)
                domain_file = None
                try:
                    for filename in os.listdir(domain_dir):
                        if filename.endswith("_domain.pddl"):
                            domain_file = os.path.join(domain_dir, filename)
                            break
                except FileNotFoundError:
                    pass

                if domain_file is None:
                    raise FileNotFoundError(
                        f"No domain file (_domain.pddl) found in directory: {domain_dir}"
                    )

                is_valid = validate_plan(domain_file, problem_path, plan_path)
                cnt += 1
                if is_valid:
                    correct += 1
                print(f"Plan {relative_path}/{base_name} valid: {is_valid}")
    print(
        f"Total plans: {cnt}, Valid plans: {correct}, Validity rate: {correct / cnt:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract regex matches from LLM generated answer files."
    )
    parser.add_argument("problem_dir", help="Problem and Domain folder path")
    parser.add_argument("plan_dir", help="Plans folder path")

    args = parser.parse_args()
    process_files(args.problem_dir, args.plan_dir)


if __name__ == "__main__":
    main()

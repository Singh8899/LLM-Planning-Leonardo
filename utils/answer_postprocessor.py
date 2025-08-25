import os
import argparse


def formatter(pattern: str) -> str:
    temp = pattern.split("assistant<|end_header_id|>")[1]
    temp = temp.strip()
    temp = temp.split("<|eot_id|>")[0]
    temp = temp.strip()
    return temp


def process_files(input_dir: str, output_dir: str) -> None:

    for root, dirs, files in os.walk(input_dir):

        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf8") as f:
                    text = f.read()

                # Get all regex matches from the file content
                match = formatter(text)

                # Preserve directory structure by calculating relative path
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                output_file = os.path.join(target_dir, file)
                with open(output_file, "w", encoding="utf8") as out_f:
                    out_f.write(match + "\n")
                print(f"Processed: {file_path} -> {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract regex matches from LLM generated answer files."
    )
    parser.add_argument(
        "input_dir", help="Input folder path containing sub-folders with .txt files"
    )
    parser.add_argument(
        "output_dir", help="Output folder path for processed answer files"
    )

    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

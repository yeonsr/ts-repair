import sys
import subprocess
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_codegen_experiment.py input.csv output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Instead of calling external API, use provided example script
    script_path = os.path.join(os.path.dirname(__file__), "example_generated_script.py")

    cmd = ["python", script_path, input_file, output_file]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
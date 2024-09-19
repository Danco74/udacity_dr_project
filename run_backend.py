import os
import subprocess
import sys
import platform

def run_backend():
    # Construct the command to run run.py with the provided arguments
    if platform.system() == "Windows":
        python_executable = os.path.join(".venv", "Scripts", "python")
    else:
        python_executable = os.path.join(".venv", "bin", "python")

    backend_script = os.path.join("app", "backend", "run.py")
    database_file = os.path.join("data", "DisasterResponse.db")
    model_file = os.path.join("model", "classifier.pkl")
    database_name = "DisasterResponse"

    # Check if the required files exist
    if not os.path.exists(database_file):
        print(f"Error: The database file '{database_file}' was not found.")
        sys.exit(1)
    if not os.path.exists(model_file):
        print(f"Error: The model file '{model_file}' was not found.")
        sys.exit(1)

    # Command to run the backend script with arguments
    command = [
        python_executable, 
        backend_script, 
        database_file, 
        database_name, 
        model_file
    ]

    print(f"Running backend: {' '.join(command)}")
    subprocess.check_call(command)

if __name__ == "__main__":
    run_backend()

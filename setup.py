import os
import subprocess
import sys
import platform

def create_virtualenv():
    # Check if the virtual environment already exists as '.venv'
    venv_dir = os.path.join(os.getcwd(), ".venv")
    
    if os.path.exists(venv_dir):
        print("Virtual environment already exists. Skipping virtual environment creation.")
    else:
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])

def install_requirements():
    # Install dependencies using pip
    if platform.system() == "Windows":
        pip_executable = os.path.join(".venv", "Scripts", "pip")
    else:
        pip_executable = os.path.join(".venv", "bin", "pip")

    print("Installing Python dependencies...")
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])

def install_frontend_dependencies():
    # Determine the npm command based on the platform
    if platform.system() == "Windows":
        npm_executable = "npm.cmd"
    else:
        npm_executable = "npm"

    # Run npm install in the frontend folder
    frontend_dir = os.path.join(os.getcwd(), "app", "frontend")
    
    if os.path.exists(frontend_dir):
        print(f"Installing frontend dependencies with {npm_executable}...")
        subprocess.check_call([npm_executable, "install"], cwd=frontend_dir)
    else:
        print(f"Error: Frontend folder '{frontend_dir}' not found.")
        sys.exit(1)

def run_data_processing():
    # Construct the command to run process_data.py with the provided arguments
    if platform.system() == "Windows":
        python_executable = os.path.join(".venv", "Scripts", "python")
    else:
        python_executable = os.path.join(".venv", "bin", "python")

    process_data_script = os.path.join("data", "process_data.py")
    messages_file = os.path.join("data", "messages.csv")
    categories_file = os.path.join("data", "categories.csv")
    database_file = os.path.join("data", "DisasterResponse.db")

    # Command to run the script with arguments
    command = [
        python_executable, 
        process_data_script, 
        messages_file, 
        categories_file, 
        database_file
    ]

    print(f"Running data processing script: {' '.join(command)}")
    subprocess.check_call(command)

def train_model():
    # Construct the command to run train_classifier.py with the provided arguments
    if platform.system() == "Windows":
        python_executable = os.path.join(".venv", "Scripts", "python")
    else:
        python_executable = os.path.join(".venv", "bin", "python")

    train_script = os.path.join("model", "train_classifier.py")
    database_file = os.path.join("data", "DisasterResponse.db")
    model_file = os.path.join("model", "classifier.pkl")

    # Command to run the script with arguments
    command = [
        python_executable, 
        train_script, 
        database_file, 
        model_file
    ]

    print(f"Running model training script: {' '.join(command)}")
    subprocess.check_call(command)

def main():
    create_virtualenv()
    
    install_requirements()

    install_frontend_dependencies()

    # Check if the database already exists
    db_file = os.path.join("data", "DisasterResponse.db")
    if os.path.exists(db_file):
        user_input = input("The database file 'DisasterResponse.db' already exists. Do you want to reprocess the data? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Running data processing...")
            run_data_processing()
        else:
            print("Skipping data processing.")
    else:
        print("Database file not found. Running data processing...")
        run_data_processing()

    # Check if the model already exists
    model_file = os.path.join("model", "classifier.pkl")
    if os.path.exists(model_file):
        user_input = input(f"The model file '{model_file}' already exists. Do you want to retrain the model? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Running model training...")
            train_model()
        else:
            print("Skipping model training.")
    else:
        print("Model file not found. Running model training...")
        train_model()
    
    print("Setup complete!")

if __name__ == "__main__":
    main()

import os
import subprocess
import sys

def run_frontend():
    # Run 'npm run dev' in the frontend folder
    frontend_dir = os.path.join(os.getcwd(), "app", "frontend")

    if not os.path.exists(frontend_dir):
        print(f"Error: Frontend folder '{frontend_dir}' not found.")
        sys.exit(1)

    print("Starting frontend development server with 'npm run dev'...")
    subprocess.check_call(["npm.cmd", "run", "dev"], cwd=frontend_dir)

if __name__ == "__main__":
    run_frontend()

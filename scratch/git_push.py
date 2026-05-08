import subprocess
import os

def run_git():
    cwd = r"c:\Users\dinhl\Downloads\Rehab-AI-Monitor"
    commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", "Update documentation to v3.0 Finalized"],
        ["git", "push", "origin", "main"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            # Continue anyway as some commits might fail if nothing to commit

if __name__ == "__main__":
    run_git()

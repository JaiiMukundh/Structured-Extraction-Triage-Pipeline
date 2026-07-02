import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent

commands = [
    
    # Evaluation & Reports
    (BASE / "evaluation" / "compare_extraction.py", []),
    (BASE / "evaluation" / "compare_validation.py", []),
    (BASE / "evaluation" / "calculate_parameters.py", []),
]

print("=" * 80)
print("STARTING COMPLETE PIPELINE RUN FROM START TO FINISH")
print("=" * 80)

for script_path, args in commands:
    print(f"\n>>> Running: {script_path.name} {' '.join(args)}")
    print("-" * 80)
    try:
        subprocess.run([sys.executable, str(script_path)] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Script {script_path.name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

print("\n" + "=" * 80)
print("PIPELINE RUN COMPLETED SUCCESSFULLY")
print("=" * 80)

import os
import subprocess

# === CONFIGURATION ===
# Get the current script's directory (i.e., csp-main)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get to postProcessing/
POSTPROCESSING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# Set base_dir to the getEulerianFields directory
base_dir = os.path.join(POSTPROCESSING_DIR, "getEulerianFields")
analysis_script = "mycsp-analysis_parcels.py"  

# === FIND TIME FOLDERS ===
time_dirs = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d.replace('.', '', 1).isdigit()
], key=lambda x: float(x))

# === PROCESS EACH TIME STEP ===
for time_dir in time_dirs:
    folder_path = os.path.join(base_dir, time_dir)
    input_file = os.path.join(folder_path, "airCloud.dat")
    output_file = os.path.join(folder_path, "airCloud_withCSP.dat")

    if not os.path.isfile(input_file):
        print(f"?? No airCloud.dat in {time_dir}, skipping...")
        continue

    if os.path.isfile(output_file):
        print(f"? Output already exists in {time_dir}, skipping...")
        continue

    print(f"?? Processing {input_file}")
    try:
        subprocess.run(["python3", analysis_script, "-f", input_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"? Error processing {input_file}: {e}")

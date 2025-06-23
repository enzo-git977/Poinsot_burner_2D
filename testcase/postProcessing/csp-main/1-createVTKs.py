import os
import subprocess
import shutil

# Define paths
base_folder = "postProcessing/cuttingPlane"
vtkName = "yPlane1"
script_path = "mycsp-analysis.py"

output_folder = vtkName

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of time directories
time_folders = sorted(
    [d for d in os.listdir(base_folder) if d.replace(".", "").isdigit()],
    key=lambda x: float(x)
)

# Process each folder sequentially
for idx, time in enumerate(time_folders):

    vtk_file = os.path.join(base_folder, time, vtkName+".vtk")
    expected_output_file = os.path.join(base_folder, time, vtkName+".vtk_csp.vtk")
    new_output_filename = os.path.join(output_folder, vtkName+f"_{idx:03d}.vtk")

    if os.path.exists(vtk_file):
        print(f"Processing: {vtk_file}")

        # Run the script
        command = ["python3", script_path, "-f", vtk_file]
        subprocess.run(command, check=True)  # Waits until it completes

        # Check if output file exists
        if os.path.exists(expected_output_file):
            # Move and rename the file
            shutil.move(expected_output_file, new_output_filename)
            print(f"Saved: {new_output_filename}")
        else:
            print(f"Warning: Expected output {expected_output_file} not found.")
    else:
        print(f"Warning: {vtk_file} not found, skipping.")


import os
import shutil

# Files and directories to remove
targets = [
    'pyjacob.cpython-310-x86_64-linux-gnu.so',
    '__pycache__',
    'out',
    'obj',
    'build',
    'axial_data.csv',
    '*Plane'
]

def remove_target(path):
    if os.path.isfile(path):
        try:
            os.remove(path)
            print(f"Removed file: {path}")
        except Exception as e:
            print(f"Failed to remove file {path}: {e}")
    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        except Exception as e:
            print(f"Failed to remove directory {path}: {e}")
    else:
        print(f"Not found, skipped: {path}")

for target in targets:
    remove_target(target)

print("\nCleanup complete.")


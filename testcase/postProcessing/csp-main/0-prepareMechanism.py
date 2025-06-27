import os
import shutil
import subprocess
import yaml
import sys

mech_dir = 'mech'

def run_cmd(cmd, description):
    print(f"{description}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during: {description}")
        print(result.stderr)
        sys.exit(1)


# Step 1: Find the YAML file that is NOT named 'mechanism.yaml'
yaml_file = None
for f in os.listdir(mech_dir):
    if f.endswith('.yaml') and f != 'mechanism.yaml':
        yaml_file = f
        break

if not yaml_file:
    raise FileNotFoundError("No YAML file found in 'mech/' other than 'mechanism.yaml'")

yaml_path = os.path.join(mech_dir, yaml_file)

# Step 2: Run yaml2ck to generate chem.inp and therm.dat
chem_inp_path = os.path.join(mech_dir, 'chem.inp')
therm_dat_path = os.path.join(mech_dir, 'therm.dat')

run_cmd([
#    'yaml2ck', yaml_path,
    'python3', '-m', 'cantera.yaml2ck', yaml_path,
    '--mechanism', chem_inp_path,
    '--thermo', therm_dat_path,
    '--overwrite'
], "Running yaml2ck")

# Step 3: Clean chem.inp — remove everything after "REACTIONS"
with open(chem_inp_path, 'r') as f:
    lines = f.readlines()

with open(chem_inp_path, 'w') as f:
    for line in lines:
        if line.strip().startswith('REACTIONS'):
            f.write('REACTIONS\n')
        else:
            f.write(line)

# Step 4: Copy the YAML file to mech/mechanism.yaml
target_yaml_path = os.path.join(mech_dir, 'mechanism.yaml')
shutil.copy(yaml_path, target_yaml_path)

# Step 5: Uninstall Cantera
run_cmd(['pip3', 'uninstall', '-y', 'cantera'], "Uninstalling Cantera")

# Step 6: Parse YAML and extract last species from 'gas' phase
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

phases = data.get('phases')
if not phases or not isinstance(phases, list):
    raise ValueError("No valid 'phases' section in the YAML file")

gas_phase = next((p for p in phases if p.get('name') == 'gas'), None)
if not gas_phase:
    raise ValueError("No phase named 'gas' found in the YAML file")

species_list = gas_phase.get('species')
if not species_list or not isinstance(species_list, list):
    raise ValueError("Could not find species list in the 'gas' phase")

#last_species = 'N2'
last_species = species_list[-1]
print(f"Last species in gas phase: {last_species}")

# Step 7: Run pyjac code generation
run_cmd([
    'python3', '-m', 'pyjac',
    '-l', 'c',
    '-i', chem_inp_path,
    '-t', therm_dat_path,
    '-ls', last_species 
], "Generating Jacobian with pyjac")
"""
# Step 8: Run pyjac.pywrap to generate wrappers
run_cmd([
    'python3', '-m', 'pyjac.pywrap',
    '-so', 'out',
    '-l', 'c'
], "Building pyjac wrapper")

# Step 9: Reinstall Cantera
run_cmd(['pip3', 'install', 'cantera'], "Reinstalling Cantera")
"""
print("\nAll steps completed successfully.")


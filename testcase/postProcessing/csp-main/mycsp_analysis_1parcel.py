print("Starting execution")
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cantera as ct
from tqdm import tqdm
import argparse
from CSP_parcel_analyzer import CSPAnalyzer_parcel as cspAnalyzer_parcel

# Read the filename from the terminal command
parser = argparse.ArgumentParser(description='CSP analysis of one parcel')
parser.add_argument("-f", "--file_name", required=True)
args = parser.parse_args()
file_name = args.file_name

# Choose target parcel 
target_parcel_id = 499

# Load mechanism
mechanism = 'mech/mechanism.yaml'
gas = ct.Solution(mechanism, transport_model="multicomponent")

# === CREATE DATAFRAME ===
data = pd.read_csv(file_name, sep=r'\s+', comment='#', header=None)

# === EXCTRACT HEADER ===
with open(file_name, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line.lstrip('#').strip().split()
            break

data.columns = header  

# === EXCTRACT Parcel row ===  
parcel_rows = data[data['PARCEL_ID'] == target_parcel_id] # we want the info of the target_parcel_id

# === EXCTRACT TEMPERATURE AND PRESSURE of a given parcel === 
T = parcel_rows.iloc[0]['T']
p = parcel_rows.iloc[0]['p']

# === Extract species mass fractions ===
Y_list = []
for species in gas.species_names:
    if species in parcel_rows.columns:
        print("Y =", species)
        Y_list.append(parcel_rows.iloc[0][species])
    else:
        Y_list.append(0.0)

Y = np.array(Y_list)

# === Eigen decomposition ===
csp = cspAnalyzer_parcel(mechanism,T)
eigenvalues, vl, vr, wherePositive = csp.eigen_decomposition_parcel(p, T, Y)

# === Find Radical Pointers===
radPoint = csp.radical_pointer_parcel(vl,vr)

# === Find largest Radical Pointers===
radicals = ['T'] + gas.species_names
idx = np.abs(radPoint[0, :, 0]).argsort()[::-1]
largest_pointer = idx[0].item()
if wherePositive > 0.0:
    # this plus1 comes from the fact that 0 is temperature, 
    # but I want it to be 1 in the plot (and following the other species)
    largest_pointerWP = idx[0].item()+1 
print("The largest pointer is ",radicals[largest_pointer])

# === Plotting radical  pointers values and participation indexes with a barchart==

# Extract mode 0 radical pointers 
values = np.zeros(len(radicals))
values[0:-1] = np.abs(radPoint[0, :, 0].real) # last one is inert so always zero

# Plot radical pointers values
plt.figure(figsize=(12, 6))
bars = plt.bar(radicals, values, color='skyblue')

# Highlight largest one
bars[largest_pointer].set_color('orange')

# Annotate if WP is satisfied
if wherePositive > 0.0:
    plt.title(f"Radical Pointers  — WP Satisfied ", fontsize=14)
    bars[largest_pointerWP].set_color('red')
    legend_handles = [Patch(color='skyblue', label='All species'),Patch(color='orange', label='Largest'),Patch(color='red', label='WP_Largest')]
    plt.legend(handles=legend_handles, loc='upper right')
else:
    plt.title(f"Radical Pointers", fontsize=14)
    legend_handles = [Patch(color='skyblue', label='All species'),Patch(color='orange', label='Largest')]
    plt.legend(handles=legend_handles, loc='upper right')

plt.xlabel('Species')
plt.ylabel('Radical Pointer Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Get participation indexes values
api,reaction_names = csp.participation_index_parcel(p, T, Y, vr)
pi = np.zeros(len(radicals))
pi[0:-1] = np.abs(api[0, 0, :].real)

# Plot participation indexes 
plt.figure(figsize=(12, 6))

bars = plt.bar(radicals, pi, color='skyblue')
plt.title(f"Participation indexes of Reaction : {reaction_names[0]}", fontsize=14)
plt.xlabel('Species')
plt.ylabel('PI values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.show()


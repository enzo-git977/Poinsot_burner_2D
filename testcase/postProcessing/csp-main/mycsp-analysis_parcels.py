import os
import numpy as np
import cantera as ct
from CSP_analyzer import CSPAnalyzer as cspAnalyzer
from tqdm import tqdm
import argparse
import pandas as pd

# === PARSE ARGS ===
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_name", required=True)
args = parser.parse_args()
file_name = args.file_name

# === LOAD MECHANISM ===
mechanism = 'mech/mechanism.yaml'
gas = ct.Solution(mechanism)

# === CREATE DATAFRAME ===
data = pd.read_csv(file_name, sep=r'\s+', comment='#', header=None)

# === EXCTRACT HEADER ===
with open(file_name, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line.lstrip('#').strip().split()
            break

data.columns = header  

# === EXCTRACT TEMPERATURE AND PRESSURE === # find with names and not columns index like previously done
T = data['T'].values
p = data['p'].values

# === Extract species mass fractions ===
Y_list = []
for species in gas.species_names:
    if species in data.columns:
        #print("Y =", species)
        Y_list.append(data[species].values)
    else:
        Y_list.append(np.zeros(len(data)))

Y = np.column_stack(Y_list)

# === CSP ANALYSIS ================================================================
csp = cspAnalyzer(mechanism, T)
eigenvalues, vl, vr, wherePositive = csp.eigen_decomposition(p, T, Y)

radPoint = csp.radical_pointer(vl, vr)

# === FIND LARGEST POINTER INDEX ===
largest_pointer = []
largest_pointerWP = np.zeros(len(T), dtype=np.int32)

for k in tqdm(range(len(T))):
    idx = np.abs(radPoint[k, :, 0]).argsort()[::-1]
    largest_pointer.append(idx[0])
    if wherePositive[k] > 0:
        largest_pointerWP[k] = idx[0] + 1  # Offset for plotting conventions

largest_pointer = np.array(largest_pointer, dtype=np.int32)

# === SAVE TO NEW FILE ===
out_file = file_name.replace(".dat", "_withCSP.dat")
augmented = np.column_stack((data, largest_pointer, wherePositive, largest_pointerWP))
header = "# PROC_ID PARCEL_ID CellI X Y Z p Ux Uy Uz T C3H8 CO2 H2O N2 O2 largest_pointer wherePositive largest_pointerWP"  # Fill with actual column names
np.savetxt(out_file, augmented, fmt="%.6e", delimiter=" ", header=header, comments='')

print(f"? CSP analysis completed. Output written to {out_file}")

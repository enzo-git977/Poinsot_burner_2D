import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri 
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
import cantera as ct

import re
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 18)

import os, sys, subprocess
import cantera as ct

from scipy.linalg import eig
from tqdm import tqdm

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa

import argparse

from pypalettes import load_cmap
cmap = load_cmap("Bay")

from CSP_analyzer import CSPAnalyzer as cspAnalyzer
from plotter import CSPPlotter 


### DATI

vtk_file = "postProcessing/cuttingPlane/0.0027/xPlane.vtk"
file_path = "axial_data.csv"
mechanism = "mech/mechanism.yaml"

T_adiab = 2328

point1 = -0.0037
point2 =  0.0012
point3 =  0.0018

locLab = ["lower right", "lower right", "lower right"]
pointLab = [r"$\bigcirc$", r"$\bigtriangleup$", r"$\square$"]






# Drop rows that contain any NaN values
df = pd.read_csv(file_path)
df_clean = df.dropna()
# Convert to numpy array if needed
data_array = df_clean.to_numpy()


# Extract relevant columns
x = data_array[:, 0]
p = data_array[:, 1]
T = data_array[:, 2] 

# Load the mechanism
gas = ct.Solution(mechanism)
species_order = gas.species_names


Y = df_clean[species_order].to_numpy()
Qdot = df_clean["Qdot"].to_numpy()


# Find the closest index to the specified point1
closest_idx1 = np.argmin(np.abs(x - point1))
x_closest1 = x[closest_idx1]
T_closest1 = T[closest_idx1]

closest_idx2 = np.argmin(np.abs(x - point2))
x_closest2 = x[closest_idx2]
T_closest2 = T[closest_idx2]

closest_idx3 = np.argmin(np.abs(x - point3))
x_closest3 = x[closest_idx3]
T_closest3 = T[closest_idx3]


csp = cspAnalyzer(mechanism,x)
eigenvalues, vl, vr, _ = csp.eigen_decomposition(p, T, Y)
radPoint = csp.radical_pointer(vl,vr)
api = csp.participation_index(p, T, Y, vr)
df = csp.explosive_dataframe(radPoint, api, eigenvalues, x, T, n_diag=5)

pd.set_option('display.max_columns', None)

print(df[0+5*closest_idx1:5+5*closest_idx1])
print(df[0+5*closest_idx2:5+5*closest_idx2])
print(df[0+5*closest_idx3:5+5*closest_idx3])


### figures

plotter = CSPPlotter(
    vtk_file=vtk_file,
    p=p, Y=Y, x=x,
    x_indices=[closest_idx1, closest_idx2, closest_idx3],
    eigenvalues=eigenvalues,
    vl=vl
)

'''
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
cf = plotter.plot_temperature_field(ax, 0.005, [-0.025,0.025], [-0.010,0.040])
fig.colorbar(cf, ax=ax, label='Temperature [K]')
plt.tight_layout()
plt.show()
'''

fig, axs = plt.subplots(2, 3, figsize=(20, 8), constrained_layout=True)
im = plotter.plot_temperature_field(fig, axs[0,0],-0.05, [-0.0201,0.0201], [-0.0702,0.002], zoomFlag=True)
fig.colorbar(im, ax=axs[0,0], orientation='vertical', label='Temperature [K]')

plotter.plot_temperature_profile_with_shading(axs[0,1], x, T/T_adiab, Qdot, xLimits=[-0.01, 0.01])

# Stoichiometric equivalence ratio
nu = 2
MM_fuel = 16.04  # g/mol (CH4)
MM_oxidant = 32.00  # g/mol (O2)
stoich_equivalenceRatio = MM_fuel / (nu * MM_oxidant)  # = 0.251

# Extract fuel and oxidant mass fractions
Y_fuel = df_clean["CH4"].to_numpy()
Y_oxidant = df_clean["O2"].to_numpy()

# Compute equivalence ratio Phi
Phi = (Y_fuel / Y_oxidant) / stoich_equivalenceRatio
ax1_twin = axs[0,1].twinx()  # Create secondary y-axis
ax1_twin.plot(x, Phi, '--', color="sienna", linewidth=0.8)
ax1_twin.set_ylabel(r'Equivalence Ratio $\phi$ [-]', color='sienna')
ax1_twin.set_ylim([0,1])
ax1_twin.tick_params(axis='y', labelcolor='sienna')


#plotter.plot_violin_eigenvalues(axs[0,2], x, eigenvalues)
plotter.plot_eigenvalue_evolution(axs[0,2], x, eigenvalues)
plotter.plot_participation_indices(axs[1,0], df, closest_idx1, pointLab[0], locLab[0])
plotter.plot_participation_indices(axs[1,1], df, closest_idx2, pointLab[1], locLab[1])
plotter.plot_participation_indices(axs[1,2], df, closest_idx3, pointLab[2], locLab[2])

plt.show()
#plt.savefig("combined_figure.pdf", bbox_inches="tight")





'''
###################################
# Chemical chain path
###################################
from subprocess import run
from pathlib import Path
import cantera as ct
import graphviz

r = ct.IdealGasReactor(gas)
net = ct.ReactorNet([r])
T = r.T
while T < 1900:
    net.step()
    T = r.T

element = 'H'  # Modify as needed

diagram = ct.ReactionPathDiagram(gas, element)
diagram.font = "CMU Serif"
diagram.show_details = False
diagram.scale = -1
diagram.threshold = 0.01  # Minimum flux to be displayed
diagram.dot_options = '' '
    node [fontsize=14, shape="box", style="filled", fillcolor="white"];
    graph [rankdir=LR, nodesep=0.3, ranksep=0.6, bgcolor=transparent];

'' '
diagram.title = 'Reaction path diagram following {0}'.format(element)

dot_file = "rxnpath.dot"
pdf_file = "rxnpath.pdf"
pdf_path = Path.cwd().joinpath(pdf_file)

diagram.write_dot(dot_file)
print(diagram.get_data())

import re

scaling_factor = 0.4  # Adjust this to control thickness scaling

def scale_value(match):
    """Scales numerical values for penwidth and arrowsize while keeping proportions."""
    original_value = float(match.group(1))
    new_value = max(original_value * scaling_factor, 0.1)  # Ensure lines remain visible
    return f"{match.group(0)[:match.start(1)-match.start()]}{new_value:.2f}"

# Read DOT file
with open(dot_file, "r") as file:
    dot_text = file.read()

# Scale penwidth and arrowsize (affects edges only)
dot_text = re.sub(r'penwidth=(\d+(\.\d+)?)', scale_value, dot_text)
dot_text = re.sub(r'arrowsize=(\d+(\.\d+)?)', scale_value, dot_text)
dot_text = re.sub(r', 0.9\"', ', 0.01\"',dot_text)

# Change only edge colors (without affecting nodes)
dot_text = re.sub(r'(edge \[.*?color=)"[^"]+"', r'\1"gray50"', dot_text, flags=re.DOTALL)

# Write the modified DOT file
with open(dot_file, "w") as file:
    file.write(dot_text)

# Generate the PDF
from subprocess import run
run(f"dot {dot_file} -Tpdf -o{pdf_file} -Gdpi=200".split())
'''

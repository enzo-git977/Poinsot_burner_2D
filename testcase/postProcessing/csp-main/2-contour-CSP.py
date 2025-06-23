import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import re

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 18)


# Folder containing VTK files
folder_path = "zPlane"
plane = "xy"

contour_level = 1e6

# CH4 settings - only plot between these values
ch4_min = 0.0010
ch4_max = 0.0050

highlight_colors = [
    "#E76254FF", 
    "#EF8A47FF", 
    "#FFD06FFF", 
    "#FFE6B7FF", 
    "#AADCE0FF",
    "#72BCD5FF", 
    #"#528FADFF", 
    "#376795FF", 
    "green" #which is better than "#1E466EFF"
]

selected_species = ["H2", "O2", "HO2", "H2O2", "CH4", "CH3", "CH2O", "C2H6"]



####################################################
# PREPARING THE PLOT

import cantera as ct

# Load the mechanism
gas = ct.Solution("mech/mechanism.yaml")

# Get species names and prepend "Non-explosive" and "Temperature"
species_list = ["Non-explosive", "Temperature"] + gas.species_names

print("Species list:")
for i, name in enumerate(species_list):
    print(f"{i:2d}: {name}")

newGray = "whitesmoke"

# ---- Build color map ----
cmap_list = []
for name in species_list:
    if name == "Non-explosive":
        cmap_list.append("white")
    elif name == "Temperature":
        cmap_list.append("black")
    elif name in selected_species:
        idx = selected_species.index(name)
        cmap_list.append(highlight_colors[idx])
    else:
        cmap_list.append(newGray)

# Ensure output folders exist
os.makedirs(os.path.join(folder_path, "whereActive"), exist_ok=True)
os.makedirs(os.path.join(folder_path, "radicalPointers"), exist_ok=True)

# Get sorted list of all VTK files
vtk_files = sorted([f for f in os.listdir(folder_path) if f.startswith(folder_path+"_") and f.endswith(".vtk")],
                   key=lambda x: int(re.search(r'_(\d+)\.vtk', x).group(1)))

print(vtk_files)

def load_mesh(filename,plane):
    if not os.path.exists(filename):
        return None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()

    # Extract triangulation information
    cells = data.GetPolys()
    points = data.GetPoints()

    # Get the number of points
    npts = points.GetNumberOfPoints()

    # Get the connectivity information
    conn = []
    cells.InitTraversal()
    idList = vtk.vtkIdList()
    while cells.GetNextCell(idList):
        numIds = idList.GetNumberOfIds()
        if numIds == 3:
            conn.append([idList.GetId(i) for i in range(3)])
        elif numIds > 3:
            # Create triangles for polygons with more than three vertices
            for i in range(numIds - 2):
                conn.append([idList.GetId(0), idList.GetId(i + 1), idList.GetId(i + 2)])

    # Convert to numpy array
    conn_array = np.array(conn)

    if len(conn_array) == 0:
        print("No valid cells found.")
        return None

    # Extract point coordinates
    x = np.zeros(npts)
    y = np.zeros(npts)
    z = np.zeros(npts)

    for i in range(npts):
        pt = points.GetPoint(i)
        x[i] = pt[0]
        y[i] = pt[1]
        z[i] = pt[2]

    # Create Triangulation directly from connectivity information
    triang = mtri.Triangulation(y, z, triangles=conn_array)
    
    if (plane == "xy"):
        triang = mtri.Triangulation(x, y, triangles=conn_array)
    elif (plane == "xz"):
        triang = mtri.Triangulation(x, z, triangles=conn_array)

    # reduce the occupied space by putting to none the unused elements
    return x, y, z, conn_array, triang


def load_scalar(filename, field):
    if not os.path.exists(filename):
        return None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()

    # Map data: cell -> point
    mapper = vtk.vtkCellDataToPointData()
    mapper.AddInputData(data)
    mapper.Update()
    mapped_data = mapper.GetOutput()

    # Extract interpolated point data
    udata = mapped_data.GetPointData().GetArray(field)

    nvls = udata.GetNumberOfTuples()

    T = np.zeros(nvls)

    for i in range(nvls):
        fieldVec = udata.GetTuple(i)
        T[i] = fieldVec[0]

    return T





cmap = mcolors.ListedColormap(cmap_list)

# Function to format species names with LaTeX subscripts
def format_species_name(name):
    return re.sub(r'(\d+)', r'$_{\1}$', name)

# Iterate over all VTK files
for file in vtk_files:
    file_path = os.path.join(folder_path, file)
    print(f"Processing {file_path}")

    # Load mesh
    mesh_data = load_mesh(file_path,plane)
    if mesh_data is None:
        continue

    x, y, z, tri, triang = mesh_data

    # Load scalar fields
    scalarField = load_scalar(file_path, "largest_pointerWP")
    Qdot = load_scalar(file_path, "Qdot")
    ch4_data = load_scalar(file_path, "CH4")

    # Create first plot (Binary Field + Qdot or CH4 Overlay)
    fig, ax = plt.subplots()
    #im = ax.tricontourf(triang, scalarField, cmap="binary")

    ax.tripcolor(triang, scalarField, cmap="binary")

    # Add flame front as a red contour line for Qdot = 1e9
    #ax.tricontour(triang, Qdot, colors = "red", levels=[contour_level],linewidth=2)

    # Overlay red patches where CH4 is in range
    valid_triangles = [t for t in triang.triangles if any(ch4_min <= ch4_data[p] <= ch4_max for p in t)]
    if valid_triangles:
        ch4_triang = mtri.Triangulation(triang.x, triang.y, triangles=np.array(valid_triangles))
        ax.tripcolor(ch4_triang, np.ones(len(ch4_triang.x)), color='red', alpha=1)

    ax.set_aspect('equal', "box")
    ax.set_xlabel("y [m]", fontsize=22)
    ax.set_ylabel("z [m]", fontsize=22)
    
    if (plane == "xy"):
        ax.set_xlabel("x [m]", fontsize=22)
        ax.set_ylabel("y [m]", fontsize=22)
    elif (plane == "xz"):
        ax.set_xlabel("x [m]", fontsize=22)
        ax.set_ylabel("z [m]", fontsize=22)

    
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    #plt.show()
    plt.savefig(f"{folder_path}/whereActive/{file.replace('.vtk', '.png')}",bbox_inches="tight")
    plt.close()

    # Ensure "Others" appears only once in the legend
    legend_labels = []
    legend_patches = []
    seen_gray = False  # Track if "Others" is already added

    for species, color in zip(species_list, cmap_list):
        formatted_species = format_species_name(species)  # Apply LaTeX formatting

        if color == newGray:
            if not seen_gray:
                legend_labels.append("Others")  # Add "Others" only once
                legend_patches.append(mpatches.Patch(color=newGray, label="Others"))
                seen_gray = True
        else:
            legend_labels.append(formatted_species)
            legend_patches.append(mpatches.Patch(color=color, label=formatted_species))
    # Create colormap
    cmap = mcolors.ListedColormap(cmap_list)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size if needed
    #im = ax.tricontourf(triang, scalarField, cmap=cmap, 
    #    levels=np.arange(-0.5, 22.5, 1), vmin=-0.5, vmax=22.5)

    # Add flame front as a red contour line for Qdot = 1e9
    #ax.tricontour(triang, Qdot, colors = "red", levels=[contour_level],linewidth=1)

    im = ax.tripcolor(triang, scalarField, cmap=cmap, vmin=-0.5, vmax=22.5, shading='flat')

    # Overlay red patches where CH4 is in range
    valid_triangles = [t for t in triang.triangles if any(ch4_min <= ch4_data[p] <= ch4_max for p in t)]
    if valid_triangles:
        ch4_triang = mtri.Triangulation(triang.x, triang.y, triangles=np.array(valid_triangles))
        ax.tripcolor(ch4_triang, np.ones(len(ch4_triang.x)), color='red', alpha=1)

    # Add a colorbar to show the mapping of values to colors
    #cbar = fig.colorbar(im, ax=ax)  # Attach the colorbar to the axis
    #cbar.set_label("Scalar Field Value", fontsize=14)  # Label the colorbar (adjust as necessary)
    #cbar.ax.locator_params(nbins=22)

    # Create legend outside the plot
    plt.legend(handles=legend_patches, loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))

    # Formatting
    ax.set_aspect('equal', "box")
    ax.set_xlabel("y [m]", fontsize=22)
    ax.set_ylabel("z [m]", fontsize=22)
    
    if (plane == "xy"):
        ax.set_xlabel("x [m]", fontsize=22)
        ax.set_ylabel("y [m]", fontsize=22)
    elif (plane == "xz"):
        ax.set_xlabel("x [m]", fontsize=22)
        ax.set_ylabel("z [m]", fontsize=22)

    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    # Adjust layout to fit legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend

    #plt.show()
    plt.savefig(f"{folder_path}/radicalPointers/{file.replace('.vtk', '.png')}",bbox_inches="tight")

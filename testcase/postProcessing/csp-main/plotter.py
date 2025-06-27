import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri 
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
import re
import os
from collections import defaultdict
mpl.rcParams['text.usetex'] = False
#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 18)
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'


import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa

import argparse

from pypalettes import load_cmap
cmap = load_cmap("Bay")


class CSPPlotter:
    def __init__(self, vtk_file, p, Y, x, x_indices, eigenvalues, vl, cmap=mpl.cm.inferno):
        self.vtk_file = vtk_file
        self.p = p
        self.Y = Y
        self.x = x
        self.closest_indices = x_indices
        self.eigenvalues = eigenvalues
        self.vl = vl
        self.cmap = cmap

        self.plane = "xy" # change if needed

        self.mesh_data = self._load_mesh()
        self.scalarField = self._load_scalar("T")
        self.Qdot = self._load_scalar("Qdot")

        #self.pointLab = [r"$\bigcirc$", r"$\bigtriangleup$", r"$\square$"]
        self.pointLab = ['o', '^', 's']

        self.cbarTitle = 'Temperature [K]'
        self.vmin = 450
        self.vmax = 2600

        self.x_indices = x_indices
        

    def _load_mesh(self):
        if not os.path.exists(self.vtk_file):
            return None

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.vtk_file)
        reader.ReadAllVectorsOn()
        reader.Update()

        data = reader.GetOutput()
        cells = data.GetPolys()
        points = data.GetPoints()
        npts = points.GetNumberOfPoints()

        conn = []
        cells.InitTraversal()
        idList = vtk.vtkIdList()
        while cells.GetNextCell(idList):
            numIds = idList.GetNumberOfIds()
            if numIds == 3:
                conn.append([idList.GetId(i) for i in range(3)])
            elif numIds > 3:
                for i in range(numIds - 2):
                    conn.append([idList.GetId(0), idList.GetId(i + 1), idList.GetId(i + 2)])

        conn_array = np.array(conn) if conn else None
        if conn_array is None:
            print(f"No valid cells found in {self.vtk_file}.")
            return None

        x, y, z = np.zeros(npts), np.zeros(npts), np.zeros(npts)
        for i in range(npts):
            pt = points.GetPoint(i)
            x[i], y[i], z[i] = pt

        # Create Triangulation directly from connectivity information
        tri = mtri.Triangulation(z, y, triangles=conn_array)
        
        if (self.plane == "xy"):
            #print("triangulation via xy plane")
            tri = mtri.Triangulation(x, y, triangles=conn_array)
        elif (self.plane == "xz"):
            tri = mtri.Triangulation(x, z, triangles=conn_array)

        return x, y, z, conn_array, tri, data

    def _load_scalar(self, field):
        if not os.path.exists(self.vtk_file):
            return None

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.vtk_file)
        reader.ReadAllVectorsOn()
        reader.Update()

        data = reader.GetOutput()
        mapper = vtk.vtkCellDataToPointData()
        mapper.AddInputData(data)
        mapper.Update()
        mapped_data = mapper.GetOutput()

        udata = mapped_data.GetPointData().GetArray(field)
        if udata is None:
            print(f"Warning: Field '{field}' not found in {self.vtk_file}.")
            return None

        return np.array([udata.GetTuple(i)[0] for i in range(udata.GetNumberOfTuples())])


    def insertZoom(self,fig,lineposition,triang,xlim_zoom,ylim_zoom):

        ## Create inset axes **outside** the figure
        ax_inset = fig.add_axes([0.001, 0.595, 0.2, 0.2]) # [l, b, w, h] in fig coordinates
        ax_inset.axhline(y=lineposition, color='white', linestyle='-', linewidth=0.75)

        ax_inset.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax_inset.tripcolor(triang, self.scalarField, cmap=self.cmap, 
            shading='gouraud', vmin=self.vmin, vmax=self.vmax)

        for k, i in enumerate(self.closest_indices):
            ax_inset.plot(self.x[i], lineposition, marker=self.pointLab[k], 
                color='white', markersize=12)

        ax_inset.set_xlim(xlim_zoom)
        ax_inset.set_ylim(ylim_zoom)
        ax_inset.set_aspect('equal')



    def plot_temperature_field(self, fig, ax, lineposition, xLimits, yLimits, zoomFlag=False):
        _, _, _, _, triang, _ = self.mesh_data
        cf = ax.tripcolor(triang, self.scalarField, cmap=self.cmap, shading='gouraud', vmin=self.vmin, vmax=self.vmax)
        
        for k, i in enumerate(self.closest_indices):
            ax.plot(self.x[i], lineposition, marker=self.pointLab[k], color='white', markersize=12)

        ax.axhline(y=lineposition, color='white', linestyle='-', linewidth=2)

        ax.set_aspect('equal', "box")

        if zoomFlag==True:
            # Define zoom-in region
            xlim_zoom = (-0.005, 0.005)
            ylim_zoom = (-0.055, -0.045)

            self.insertZoom(fig,lineposition,triang,xlim_zoom,ylim_zoom)
            ax.plot(
                [xlim_zoom[0], xlim_zoom[1], xlim_zoom[1], xlim_zoom[0], xlim_zoom[0]],
                [ylim_zoom[0], ylim_zoom[0], ylim_zoom[1], ylim_zoom[1], ylim_zoom[0]],
                color="white", linestyle="--", linewidth=1
            )

        ax.set_xlabel("z [m]", fontsize=22)
        ax.set_ylabel("y [m]", fontsize=22)
        
        if (self.plane == "xy"):
            ax.set_xlabel("x [m]", fontsize=22)
            ax.set_ylabel("y [m]", fontsize=22)
        elif (self.plane == "xz"):
            ax.set_xlabel("x [m]", fontsize=22)
            ax.set_ylabel("z [m]", fontsize=22)

        ax.set_xlim(xLimits)
        ax.set_ylim(yLimits)
        return cf


    def plot_temperature_profile_with_shading(self, ax, xLine, TLine, QdotLine, threshold=3e8,xLimits=[-0.025, 0.025]):
        # Identify shaded regions where Qdot > threshold
        below_threshold = QdotLine > threshold
        starts = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1  
        ends = np.where(np.diff(below_threshold.astype(int)) == -1)[0]  

        if below_threshold[0]:  
            starts = np.insert(starts, 0, 0)
        if below_threshold[-1]:  
            ends = np.append(ends, len(self.x) - 1)

        # Plot shaded regions
        for start, end in zip(starts, ends):
            ax.axvspan(xLine[start], xLine[end], color='lightgray', alpha=0.5)

        # Plot temperature profile
        ax.plot(xLine, TLine, '-', color='k')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('Temperature [K]', color='k')
        ax.tick_params(axis='y', labelcolor='k')

        # Mark investigated points
        for i, idx in enumerate(self.x_indices):
            marker = ['o', '^', 's'][i]
            ax.scatter(xLine[idx], TLine[idx], edgecolor='royalblue', facecolor='none',
                       s=100, marker=marker, linewidth=1.5)
        
        ax.set_xlim(xLimits)

        return ax


    def plot_eigenvalue_evolution(self, ax, x, eigenvalues, tol=1e-5):
        """
        Plots the real part of eigenvalues along the axial line, excluding modes that are identically zero.
        
        Parameters:
        - ax: matplotlib axis
        - x: 1D array of axial positions
        - eigenvalues: 2D array of shape (len(x), n_modes), complex values
        - tol: numerical tolerance to consider eigenvalues as zero
        """
        n_modes = eigenvalues.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_modes))

        for mode_idx in range(n_modes):
            real_vals = eigenvalues[:, mode_idx].real
            print(max(real_vals),min(real_vals))
            if np.all(np.abs(real_vals) < tol):
                print("skipping")
                continue  # skip conservation mode (zero everywhere)

            ax.plot(x, real_vals, label=f"Mode {mode_idx+1}", color=colors[mode_idx])

        ax.set_xlabel("x [m]")
        ax.set_xticks([-0.02,0,0.02])
        ax.set_ylabel(r"Re$(\lambda_i)$ [1/s]")
        ax.set_yscale("symlog")
        ax.set_ylim([-1e10,1e6])

        # Only horizontal grid lines
        ax.grid(True, which='both', axis='y')
        ax.grid(False, axis='x')

        # Set y-ticks manually
        yticks = [1e4, 1e1, -1e1, -1e4, -1e7]
        ax.set_yticks(yticks)
        ax.set_yticklabels([r"$10^{4}$", r"$10^{1}$", r"$-10^{1}$", r"$-10^{4}$", r"$-10^{7}$"])

        # Optional: enable legend if needed
        # ax.legend(title="Eigenmodes", fontsize=10)



    '''
    def plot_participation_indices(self, ax, df, index, point_label, location):
    
        from pypalettes import load_cmap
        cmap = load_cmap("Bay")

        # Extract top 5 reactions and their participation index values
        def format_reaction_equation(equation):
            # Convert reaction arrows
            equation = equation.replace("<=>", "$\\rightleftharpoons$")\
                               .replace("=>", "$\\rightarrow$")\
                               .replace("<=", "$\\leftarrow$")  
            
            # Add subscripts only for numbers that follow species (e.g., O2 -> O$_2$)
            equation = re.sub(r'(?<=[A-Za-z])(\d+)', r'$_\1$', equation)  

            # Add subscripts to leading stoichiometric coefficients (avoid for species names)
            equation = re.sub(r'(\s)(\d+)(?=[A-Za-z])', r'\1$_\2$', equation)

            return equation

        
        data_restricted = df[0+5*index:5+5*index]

        reaction_labels_raw = data_restricted['amp_reac'].tolist()
        reaction_labels = [format_reaction_equation(r) for r in reaction_labels_raw]        
        participation_values = data_restricted['amp_par_idx'].tolist()

        y_pos = np.arange(len(reaction_labels))
        ax.barh(y_pos, participation_values, align='center', color=cmap.colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(reaction_labels, fontsize=13)
        ax.invert_yaxis()
        ax.set_xlabel('CSP Participation Index [-]')
        ax.legend(handles=[], frameon=False, loc=location, title=f"Point {point_label}", title_fontsize=11)
    '''
    
    def plot_participation_indices(self, ax, df, index, point_label, location):
        from pypalettes import load_cmap
        cmap = load_cmap("Bay")

        # Extract top 5 reactions and their participation index values
        data_restricted = df[0 + 5 * index: 5 + 5 * index]

        reaction_labels = data_restricted['amp_reac_num'].tolist()  # Use number with (f)/(b)
        participation_values = data_restricted['amp_par_idx'].tolist()

        y_pos = np.arange(len(reaction_labels))
        ax.barh(y_pos, participation_values, align='center', color=cmap.colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(reaction_labels, fontsize=18)
        ax.invert_yaxis()
        ax.set_xlabel('CSP Participation Index [-]')
        ax.legend(handles=[], frameon=False, loc=location, title=f"Point {point_label}", title_fontsize=17)
        
    
    def find_boundary_edges(self,triangles):
        # Function to find the boundary edges from a NumPy array of triangles
        edge_count = defaultdict(int)
        for triangle in triangles:
            # For each triangle, get its three edges
            edges = [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0])
            ]
            # Sort each edge's nodes so (a, b) and (b, a) are counted as the same
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                edge_count[sorted_edge] += 1
        # Boundary edges are those that appear only once in the edge count
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return boundary_edges
        
    
    def plot_geom(self, ax, plane="xy", line_color='black', line_width=0.8):
        #Plot mesh boundary edges on a given matplotlib axis.
        x, y, z, conn_array, *_ = self.mesh_data
        boundary_edges = self.find_boundary_edges(conn_array)

        if plane == "zy":
            X, Y = z, y
        elif plane == "xz":
            X, Y = x, z
        else:  # "xy"
            X, Y = x, y

        for edge in boundary_edges:
            x_vals = [X[edge[0]], X[edge[1]]]
            y_vals = [Y[edge[0]], Y[edge[1]]]
            ax.plot(x_vals, y_vals, color=line_color, linewidth=line_width)


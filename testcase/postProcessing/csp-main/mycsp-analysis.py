import numpy as np
import os, sys, subprocess
import cantera as ct
from tqdm import tqdm
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
import argparse

from CSP_analyzer import CSPAnalyzer as cspAnalyzer

# Read the filename from the terminal command
parser = argparse.ArgumentParser(description='CSP analysis of the vtk slices')
parser.add_argument("-f", "--file_name", required=True)
args = parser.parse_args()
file_name = args.file_name

mechanism = 'mech/mechanism.yaml'
gas = ct.Solution(mechanism, transport_model="multicomponent")

def load_scalar(vtk_file,field):
    if not os.path.exists(vtk_file):
        return None

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()
    mapper = vtk.vtkCellDataToPointData()
    mapper.AddInputData(data)
    mapper.Update()
    mapped_data = mapper.GetOutput()

    udata = mapped_data.GetPointData().GetArray(field)
    if udata is None:
        print(f"Warning: Field '{field}' not found in {vtk_file}.")
        return None

    return np.array([udata.GetTuple(i)[0] for i in range(udata.GetNumberOfTuples())])


def slice_reader(file_name, gas):
    """
    Reads a VTK slice file and extracts temperature, pressure, and mass fractions.
    """
    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.ReadAllVectorsOn()
    reader.Update()
    vtk_data = reader.GetOutput()
    
    # Extract field names
    field_names = [vtk_data.GetPointData().GetArrayName(i) for i in range(vtk_data.GetPointData().GetNumberOfArrays())]
    
    # Read temperature and pressure
    T = load_scalar(file_name, 'T') if 'T' in field_names else None
    P = load_scalar(file_name, 'p') if 'p' in field_names else None

    # Extract species mass fractions
    Y_list = []
    for species in gas.species_names:
        print("Y =", species)
        if species in field_names:
            Y_list.append(load_scalar(file_name, species))
        else:
            Y_list.append(np.zeros_like(T))  # Default to zero if species is missing
    
    Y = np.column_stack(Y_list)

    return vtk_data, T, P, Y, field_names


vtk_data, T, p, Y, field_names = slice_reader(file_name, gas)





###############
csp = cspAnalyzer(mechanism,T)
eigenvalues, vl, vr, wherePositive = csp.eigen_decomposition(p, T, Y)

# Find Radical Pointers
radPoint = csp.radical_pointer(vl,vr)


# for all points, find the most relevant (largest) radical pointer
radicals = ['T'] + gas.species_names
largest_pointer = []
largest_pointerWP = np.zeros(len(T))

for k in tqdm(range(len(T))):
    idx = np.abs(radPoint[k, :, 0]).argsort()[::-1]
    largest_pointer.append(idx[0].item())
    if wherePositive[k] > 0:
        # this plus1 comes from the fact that 0 is temperature, 
        # but I want it to be 1 in the plot (and following the other species)
        largest_pointerWP[k] = idx[0].item()+1 


# Make the new vtk file
vtk_csp_data = dsa.WrapDataObject(vtk_data)

# Append only Radical Pointers for the explosive mode.
# Can be modified to include all the modes but the
# resulting vtk file will grow in size dramatically.
for i in range(radPoint.shape[1]):
    vtk_csp_data.PointData.append(radPoint[:, i, 0].real,
                                  'p{}'.format(radicals[i]))
                                  
largest_pointer = np.array(largest_pointer, dtype=np.int32)
vtk_csp_data.PointData.append(largest_pointer, 'largest_pointer')
vtk_csp_data.PointData.append(wherePositive, 'wherePositive')
vtk_csp_data.PointData.append(largest_pointerWP, 'largest_pointerWP')


def vtk_write(vtk_data, fname):
    print('=> Writing {} file'.format(os.path.basename(fname)))
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(fname)
    writer.SetInputData(vtk_data.VTKObject)
    writer.Write()

# Write new vtk file with postprocess fields
vtk_write(vtk_csp_data, fname=file_name + '_csp.vtk')



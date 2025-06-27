import vtk
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# File path
vtk_file = "postProcessing/cuttingPlane/0.0027/xPlane.vtk"
nameOfAxialDirection = "z"

fixedAxis = "y"
position = -0.05


###############################################################

# Function to load polydata
def load_polydata(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.Update()

    data = reader.GetOutput()
    if data.GetNumberOfPoints() == 0:
        raise ValueError("VTK file contains no points!")

    return data


def extract_line(vtk_data, line_axis="z", fixed_axis="y", fixed_value=0.005, tolerance=1e-7):
    """
    Extract a 1D line along `line_axis` at a fixed position along `fixed_axis`.
    
    Args:
        vtk_data: loaded VTK dataset
        line_axis: 'x', 'y', or 'z' (the axis along which values are sorted and extracted)
        fixed_axis: 'x', 'y', or 'z' (the axis used to filter the slice)
        fixed_value: value at which to fix the `fixed_axis`
        tolerance: numerical tolerance to match the fixed position

    Returns:
        dict with keys: line_axis, field1, field2, ..., each as numpy arrays
    """

    axis_map = {"x": 0, "y": 1, "z": 2}
    if line_axis not in axis_map or fixed_axis not in axis_map:
        raise ValueError("line_axis and fixed_axis must be 'x', 'y', or 'z'")
    if line_axis == fixed_axis:
        raise ValueError("line_axis and fixed_axis must be different")

    i_line = axis_map[line_axis]
    i_fixed = axis_map[fixed_axis]

    points = vtk_data.GetPoints()
    num_points = points.GetNumberOfPoints()

    selected = []
    for i in range(num_points):
        coords = points.GetPoint(i)
        if abs(coords[i_fixed] - fixed_value) < tolerance:
            selected.append((coords[i_line], i))  # (line coordinate, point index)

    if not selected:
        raise ValueError(f"No points found near {fixed_axis} = {fixed_value} within tolerance {tolerance}")

    selected.sort()  # sort by line_axis coordinate

    # Extract all scalar point data
    field_data = {}
    for i in range(vtk_data.GetPointData().GetNumberOfArrays()):
        array = vtk_data.GetPointData().GetArray(i)
        field_name = array.GetName()
        values = [array.GetTuple1(idx) for _, idx in selected]
        field_data[field_name] = np.array(values)

    structured_data = {
        line_axis: np.array([coord for coord, _ in selected]),
        **field_data
    }

    return structured_data



# Function to save extracted data as CSV
def save_to_csv(data, filename="axial_data.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved extracted data to {filename}")


# Function to plot temperature distribution
def plot_temperature(data,nameOfAxialDirection):
    if nameOfTemperatureField not in data:
        raise KeyError("Temperature field ('T') not found in extracted data.")

    plt.figure(figsize=(6, 4))
    plt.plot(data[nameOfAxialDirection], data[nameOfTemperatureField], "k-", label="Temperature")
    plt.xlabel(f"Axial position along {nameOfAxialDirection} [m]")
    plt.ylabel("Temperature [K]")
    plt.grid("")
    plt.legend()
    plt.show()


nameOfTemperatureField = "T"

# Load and extract
polydata = load_polydata(vtk_file)
data = extract_line(polydata, line_axis=nameOfAxialDirection, fixed_axis=fixedAxis, fixed_value=position)

# Save to CSV
save_to_csv(data, filename="axial_data.csv")

plot_temperature(data,nameOfAxialDirection)




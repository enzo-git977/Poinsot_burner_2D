import os
import sys
sys.path.append("/mnt/d/csp-main")  
import pandas as pd
import matplotlib.pyplot as plt
from plotter import CSPPlotter 

# === HELPER FUNCTION ===
def shifted_points(x, y, shift_fraction=0.1):
    shifted_x = []
    shifted_y = []
    for j in range(len(x) - 1):
        dx = x[j+1] - x[j]
        dy = y[j+1] - y[j]
        # Point shifted backward from the end point by shift_fraction
        shifted_x.append(x[j+1] - dx * shift_fraction)
        shifted_y.append(y[j+1] - dy * shift_fraction)
    return shifted_x, shifted_y

# === CREATE INSTANCE ===
plotter = CSPPlotter(vtk_file=os.path.expanduser("~/OpenFOAM/maffeise-12/run/Poinsot_burner_2D/testcase/postProcessing/cuttingPlanes/0.04/zMid.vtk"), p=None, Y=None, x=None, x_indices=None, eigenvalues=None, vl=None)

# === CONFIGURATION ===
base_dir = os.path.expanduser('~/OpenFOAM/maffeise-12/run/Poinsot_burner_2D/testcase/postProcessing/getEulerianFields')
target_parcel_id = 96

# === INITIALISATION ===
x_positions = []
y_positions = []
T_positions = []
WP_flag_list= []
time_stamps = []

# === FIND TIME FOLDERS ===
time_folders = sorted(
    [f for f in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, f)) and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)

# === LOOP OVER TIME STEPS ===
for time_str in time_folders:
    file_path = os.path.join(base_dir, time_str, 'airCloud_withCSP.dat')

    if not os.path.isfile(file_path):
        continue

    try:
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        # === CREATE DATAFRAME ===
        data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, engine='python')

        # === EXCTRACT HEADER ===
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header = line.lstrip('#').strip().split()
                break

        data.columns = header  
        #print(f"Extracted header: {header}") # for debug

        if data.empty:
            continue
        
        parcel_rows = data[data['PARCEL_ID'] == target_parcel_id] # we want the info of the target_parcel_id
	
        if not parcel_rows.empty:
            # === EXCTRACT values === 
            x = parcel_rows.iloc[0]['X']
            y = parcel_rows.iloc[0]['Y']
            T = parcel_rows.iloc[0]['T']
            flag_WP = parcel_rows.iloc[0]['wherePositive'] 
            x_positions.append(x)
            y_positions.append(y)
            T_positions.append(T)
            WP_flag_list.append(flag_WP)
            time_stamps.append(float(time_str))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")c

# === SET FIXED AXIS LIMITS BASED ON ALL DATA ===
fixed_xlim = (-0.030, 0.15) # zoom in order to see better the parcels
fixed_ylim = (-0.002, 0.052)

# === PLOTTING LOOP ===
for i in range(len(x_positions)):
    #plt.figure(figsize=(6, 5))
    fig, ax = plt.subplots()  
    plotter.plot_geom(ax, plane="xy")  # overlays geometry

    # Plot previous positions in grey
    if i > 0:
        ax.plot(x_positions[:i+1], y_positions[:i+1], 'o-', color='grey', alpha=0.3)
    
    # Compute shifted arrow positions for segments up to i
        sx, sy = shifted_points(x_positions[:i+1], y_positions[:i+1], shift_fraction=0.1)

    # Plot only first and last arrow 
        if len(sx) >= 1:
            ax.plot(sx[0], sy[0], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)
            if len(sx) > 1:
                ax.plot(sx[-1], sy[-1], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)

    # Plot current position in blue
    ax.plot(x_positions[i], y_positions[i], 'o', color='blue',label=f'Parcel ID: {target_parcel_id}')
    
    # Fix axes to constant limits
    ax.set_xlim(fixed_xlim)
    ax.set_ylim(fixed_ylim)
    ax.legend(
    loc='center left', 
    bbox_to_anchor=(0.76, 0.9),
    fontsize=8,           # small font size
    handlelength=1.5,     # length of the legend line (shorter than default)
    handleheight=0.5,     # height of the legend handle
    handletextpad=0.4,    # space between handle and text
    markerscale=0.5,      # scale down marker size
    borderaxespad=0       # reduce padding between legend and axes
    )
    
    #ax.set_aspect('equal', "box") for some reason it leads to the axis changing for every timestep so comment
    
    # Flag to point out if species are explosive
    flag_text = "Explosive" if bool(WP_flag_list[i]) else "Not Explosive"
    
    # Annotate with info box and arrow 
    ax.annotate(
    f"T = {T_positions[i]:.4f} K \n{flag_text}",
    xy=(x_positions[i], y_positions[i]),              # point (target of arrow)
    xytext=(x_positions[i], y_positions[i] + 0.005),  # text location (above point)
    fontsize=8,
    ha='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6, lw=1)
    )

    # Styling 
    ax.set_title(f"Parcel Trajectory - t = {time_stamps[i]:.4f} s")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Save the image
    #plt.tight_layout()
    fig.savefig(f"pictures_csp/parcel_{i:04d}.png", dpi=150)
    plt.close(fig)
    print(f"Plotting successful for t={time_stamps[i]:.4f}")
    


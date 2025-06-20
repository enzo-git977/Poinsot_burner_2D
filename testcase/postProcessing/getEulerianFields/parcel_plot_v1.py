import os
import pandas as pd
import matplotlib.pyplot as plt

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

# === CONFIGURATION ===
base_dir = os.path.expanduser('~/OpenFOAM/maffeise-12/run/Poinsot_burner_2D/testcase/postProcessing/getEulerianFields')
target_parcel_id = 96

# === INITIALISATION ===
x_positions = []
y_positions = []
T_positions = []
time_stamps = []

# === FIND TIME FOLDERS ===
time_folders = sorted(
    [f for f in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, f)) and f.replace('.', '', 1).isdigit()],
    key=lambda x: float(x)
)


# === LOOP OVER TIME STEPS ===
for time_str in time_folders:
    file_path = os.path.join(base_dir, time_str, 'airCloud.dat')

    if not os.path.isfile(file_path):
        continue

    try:
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        df = pd.read_csv(file_path, sep=r'\s+', header=None, comment='#', engine='python')
        

        if df.empty:
            continue

        parcel_rows = df[df[1] == target_parcel_id]

        if not parcel_rows.empty:
            x = parcel_rows.iloc[0][3]
            y = parcel_rows.iloc[0][4]
            T = parcel_rows.iloc[0][10]
            x_positions.append(x)
            y_positions.append(y)
            T_positions.append(T)
            time_stamps.append(float(time_str))

    except Exception:
        pass

# === SET FIXED AXIS LIMITS ===
fixed_xlim = (-0.025, 0.3)
fixed_ylim = (0.0, 0.05)

# === PLOTTING LOOP ===
for i in range(len(x_positions)):
    plt.figure(figsize=(6, 5))

    # Plot previous positions in grey
    if i > 0:
        plt.plot(x_positions[:i+1], y_positions[:i+1], 'o-', color='grey', alpha=0.3)
    
    # Compute shifted arrow positions for segments up to i
        sx, sy = shifted_points(x_positions[:i+1], y_positions[:i+1], shift_fraction=0.1)

    # Plot only first and last arrow 
        if len(sx) >= 1:
            plt.plot(sx[0], sy[0], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)
            if len(sx) > 1:
                plt.plot(sx[-1], sy[-1], marker='>', linestyle='', color='grey', alpha=0.7, markersize=6)

    # Plot current position in blue
    plt.plot(x_positions[i], y_positions[i], 'o', color='blue')
    
    # Fix axes to constant limits
    plt.xlim(fixed_xlim)
    plt.ylim(fixed_ylim)
    plt.axis('equal')

    # Annotate with info box and arrow 
    plt.annotate(
    f"T = {T_positions[i]:.4f} K",
    xy=(x_positions[i], y_positions[i]),              # point (target of arrow)
    xytext=(x_positions[i], y_positions[i] + 0.003),  # text location (above point)
    fontsize=8,
    ha='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.6, lw=1)
    )

    # Styling 
    plt.title(f"Parcel Trajectory - t = {time_stamps[i]:.4f} s")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # Save the image
    #plt.tight_layout()
    plt.savefig(f"pictures_v1/parcel_{i:04d}.png", dpi=150)
    plt.close()
    print(f"Plotting successful for t={time_stamps[i]:.4f}")
    


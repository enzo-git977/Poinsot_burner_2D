import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = CURRENT_DIR

def find_hot_parcels(base_dir, filename="airCloud_withCSP.dat", temperature_threshold=1000.0):
    time_folders = sorted(
        [f for f in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, f)) and f.replace('.', '', 1).isdigit()],
        key=lambda x: float(x)
    )

    hot_parcels = set() # using set to avoid repetition of the same ID

    for time_str in time_folders:
        file_path = os.path.join(base_dir, time_str, filename)

        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            continue

        try:
            # Read header
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        header = line.lstrip('#').strip().split()
                        break

            # Read data
            df = pd.read_csv(file_path, sep=r'\s+', header=None, comment='#', engine='python')
            df.columns = header

            # Filter parcels above temperature threshold
            hot_df = df[df['T'] > temperature_threshold]

            if not hot_df.empty:
                hot_ids = hot_df['PARCEL_ID'].unique()
                #print(f"[{time_str}] Found hot parcels: {hot_ids}")
                hot_parcels.update(hot_ids)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if hot_parcels:
        print("\n Parcels above 1000K across time steps:")
        for pid in sorted(hot_parcels):
            print(f"Parcel ID: {pid}")
    else:
        print("No parcels found with temperature above threshold.")


find_hot_parcels(base_dir)


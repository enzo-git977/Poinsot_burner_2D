input_file = "particlePosition.txt"  
output_file = "particlePosition_fuel_entry.txt"

new_x = -0.025       # constant new x coordinate
new_y_min = 0.012    # new min y
new_y_max = 0.017    # new max y

def process_positions():
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        header_lines = []
        coords = []
        reading_coords = False

        for line in fin:
            stripped = line.strip()
            if stripped == '(':
                reading_coords = True
                continue
            if stripped == ');':
                reading_coords = False
                continue
            
            if reading_coords:
                # Extract coords from line
                line_clean = stripped.strip('()')
                x_str, y_str, z_str = line_clean.split()
                coords.append((float(x_str), float(y_str), float(z_str)))
            else:
                header_lines.append(line)

        # Write header lines before coordinates block
        for hline in header_lines:
            fout.write(hline)

        # Write coordinates block
        fout.write('(\n')

        n = len(coords)
        for i, (x, y, z) in enumerate(coords):
            # Distribute y evenly between new_y_min and new_y_max
            if n > 1:
                new_y = new_y_min + i * (new_y_max - new_y_min) / (n - 1)
            else:
                new_y = new_y_min
            fout.write(f'    ({new_x:.6f} {new_y:.6f} {z:.6f})\n')

        fout.write(');\n')

if __name__ == "__main__":
    process_positions()
    print(f"Modified particle positions saved to '{output_file}'")

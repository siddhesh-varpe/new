import numpy as np
import csv
import os
import math
import argparse # For CLI
import json     # To read openings.json

# ---------------- PARAMETERS ----------------
BRICK = (0.2, 0.1, 0.1)
WALL_THICKNESS = 0.2
MAX_BRICKS = 10_000
PLACEMENT_FILE = "placement.csv"
OPENINGS_LOG_FILE = "openings.csv" # Log file as requested

# Global lists to store data
bricks = []
brick_id = 1
# Global dimensions (set in main)
L, W, H, t = 0, 0, 0, 0

# ---------------- STEP 1 — FIND BEST SIZE ----------------
def find_best_size():
    """Finds optimal dimensions and returns them."""
    def brick_count(L, W, H):
        L_in, W_in = L - 2*WALL_THICKNESS, W - 2*WALL_THICKNESS
        if L_in <= 0 or W_in <= 0: return math.inf
        walls = 200 * (L + W) * H
        floor = 50 * L_in * W_in
        roof = floor
        return walls + floor + roof

    print("Running optimization...")
    best = None
    for L_ in np.arange(0.8, 20.1, 0.1):
        for W_ in np.arange(0.8, 20.1, 0.1):
            for H_ in np.arange(1.8, 3.1, 0.1):
                n = brick_count(L_, W_, H_)
                if n <= MAX_BRICKS:
                    L_in, W_in = L_ - 2*WALL_THICKNESS, W_ - 2*WALL_THICKNESS
                    vol = L_in * W_in * H_
                    if not best or vol > best["vol"]:
                        best = dict(L=L_, W=W_, H=H_, n=n, vol=vol)
    
    if not best:
        raise ValueError("Could not find any valid dimensions.")
        
    print(f"✅ Best design: Outer {best['L']:.1f} × {best['W']:.1f} × {best['H']:.1f} m")
    return best

# ---------------- STEP 2 — GENERATE BRICKS ----------------
def add_brick(x, y, z, orientation, active=1, region=""):
    """Add brick record with orientation + active flag."""
    global brick_id
    bricks.append([
        brick_id, round(x,3), round(y,3), round(z,3),
        *orientation, active, region
    ])
    brick_id += 1

def generate_solid_structure():
    """Generates the full, solid brick cuboid."""
    global L, W, H, t, bricks, brick_id
    bricks = []
    brick_id = 1
    
    print(f"Generating bricks for {L:.1f} × {W:.1f} × {H:.1f} m structure...")
    x_vals = np.arange(0, L, 0.1)
    y_vals = np.arange(0, W, 0.1)
    z_vals = np.arange(0, H, 0.1)
    
    for z in z_vals:
        for x in x_vals: add_brick(x, 0, z, (1,0,0), 1, "front")
        for x in x_vals: add_brick(x, W-t, z, (1,0,0), 1, "back")
    for z in z_vals:
        for y in y_vals: add_brick(0, y, z, (0,1,0), 1, "left")
        for y in y_vals: add_brick(L-t, y, z, (0,1,0), 1, "right")
    for x in np.arange(t, L-t, 0.2):
        for y in np.arange(t, W-t, 0.1):
            add_brick(x, y, 0, (0,0,1), 1, "floor")
    for x in np.arange(t, L-t, 0.2):
        for y in np.arange(t, W-t, 0.1):
            add_brick(x, y, H, (0,0,1), 1, "roof")
    print(f"✅ Bricks generated: {len(bricks)}")

# ---------------- STEP 3 — GEOMETRY API ----------------

def carve_opening(wall_name, x_mm, z_mm, width_mm, height_mm, opening_type="Opening"):
    """
    Core API: Finds and deactivates bricks inside a bounding box on a specific wall.
    Coordinates are in mm, relative to the wall's bottom-left corner.
    """
    global L, W, t, bricks
    
    # 1. Convert all inputs from mm to meters
    x_rel = x_mm / 1000.0
    z_min = z_mm / 1000.0
    width = width_mm / 1000.0
    height = height_mm / 1000.0
    z_max = z_min + height

    # 2. Define the 3D bounding box for the opening
    if wall_name == "front":
        x_min, x_max = x_rel, x_rel + width
        y_min, y_max = 0, t  # Carve through the whole wall thickness
    elif wall_name == "back":
        x_min, x_max = x_rel, x_rel + width
        y_min, y_max = W - t, W
    elif wall_name == "left":
        x_min, x_max = 0, t
        y_min, y_max = x_rel, x_rel + width # On side walls, "x" is along global Y
    elif wall_name == "right":
        x_min, x_max = L - t, L
        y_min, y_max = x_rel, x_rel + width
    else:
        print(f"Warning: Unknown wall '{wall_name}'. Skipping opening.")
        return

    print(f"  Carving {opening_type} in {wall_name} wall at x={x_mm}, z={z_mm}...")

    # 4. Iterate and deactivate bricks
    deactivated_count = 0
    for b in bricks:
        if b[8] != wall_name:  # b[8] is the 'region' column
            continue
            
        bx, by, bz = b[1:4] # Brick's (x, y, z) coordinates
        
        # Check if the brick's center is inside the bounding box
        if (x_min - 0.001 <= bx <= x_max + 0.001 and
            y_min - 0.001 <= by <= y_max + 0.001 and
            z_min - 0.001 <= bz <= z_max + 0.001):
            
            b[7] = 0  # Set active flag (index 7) to 0
            deactivated_count += 1
            
    print(f"    Done. Deactivated {deactivated_count} bricks.")

# --- Wrapper Functions ---
def carve_door(wall_name, x_mm, width_mm, height_mm):
    """Wrapper for carve_opening, setting z_mm=0."""
    carve_opening(wall_name, x_mm, 0, width_mm, height_mm, opening_type="Door")

def carve_window(wall_name, x_mm, z_mm, width_mm, height_mm):
    """Wrapper for carve_opening."""
    carve_opening(wall_name, x_mm, z_mm, width_mm, height_mm, opening_type="Window")

# ---------------- MAIN EXECUTION ----------------
def main():
    parser = argparse.ArgumentParser(description="Generate a brick cuboid from an openings file.")
    parser.add_argument(
        "--openings",
        type=str,
        required=True,
        help="Path to the openings.json file generated by the Gemini script."
    )
    args = parser.parse_args()

    # --- STEP 1: Find Best Size ---
    best = find_best_size()
    global L, W, H, t
    L, W, H = best["L"], best["W"], best["H"]
    t = WALL_THICKNESS
    
    # --- STEP 2: Generate Bricks ---
    generate_solid_structure()

    # --- STEP 3: Load & Carve Openings ---
    print(f"\n--- Loading Openings from {args.openings} ---")
    try:
        with open(args.openings, 'r') as f:
            openings_list = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {args.openings} not found.")
        print("Please run `prompt_to_openings.py` first.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: {args.openings} is not a valid JSON file.")
        return

    # --- STEP 4: Export CSVs ---
    
    # Log openings to openings.csv (as requested)
    if openings_list:
        print(f"Logging {len(openings_list)} openings to {OPENINGS_LOG_FILE}...")
        with open(OPENINGS_LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=openings_list[0].keys())
            writer.writeheader()
            writer.writerows(openings_list)
    
    # Carve openings by processing the list
    for item in openings_list:
        try:
            if item['type'] == 'door':
                carve_door(item['wall'], item['x_mm'], item['width_mm'], item['height_mm'])
            elif item['type'] == 'window':
                carve_window(item['wall'], item['x_mm'], item['z_mm'], item['width_mm'], item['height_mm'])
        except KeyError:
            print(f"Warning: Skipping malformed opening object: {item}")
            
    active_count = sum(1 for b in bricks if b[7] == 1)
    print(f"🚪 Final active bricks: {active_count} / {len(bricks)}")
    
    # Write final placement.csv
    print(f"\nWriting final brick data to {PLACEMENT_FILE}...")
    header_placement = ["id", "x", "y", "z", "dx", "dy", "dz", "active", "region"]
    with open(PLACEMENT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_placement)
        writer.writerows(bricks)
    
    print(f"📄 {PLACEMENT_FILE} written successfully.")
    print("\n✅ Process complete. You can now run render.py to visualize.")

if __name__ == "__main__":
    main()
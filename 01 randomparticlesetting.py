#01randomparticlesetting.py
import csv
import random
import numpy as np
import os


# == NUMBER OF PARTICLES TO SHAKE IN CONTAINER == 
n_falling = 264  

os.chdir("/Users/liliy/Documents/GitHub/ISS2.0/data/") # Change accordingly

def WRITE(file, data):
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"data written to '{file}'")

def WRITE_DICT(file, data_dict):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_dict.keys())   # Header
        writer.writerow(data_dict.values()) # Values
        print(f"box info written to '{file}'")


# === BOX DIMENSIONS === (saved in .csv file)

# Box dimensions (must match simulation!)
box_left = 0.01
box_right = 0.19
box_bottom = 0.01
box_top = 0.19

box_width = box_right - box_left   # 0.18m
box_height = box_top - box_bottom  # 0.18m


# ===== TRIAL RANGES =====
# Particle size ranges (3 distinct sizes). 

# Example: 
# r_small = (0.003, 0.004)    # Small: 3-4mm
# r_medium = (0.0045, 0.0055) # Medium: 4.5-5.5mm
# r_large = (0.006, 0.007)    # Large: 6-7mm

#01
# r_small = (0.0048, 0.00492)    
# r_medium = (0.00593, 0.00508) 
# r_large = (0.00509, 0.00520) 

#02
# r_small = (0.0046, 0.00484)    
# r_medium = (0.00485, 0.00516) 
# r_large = (0.00517, 0.00540) 

# #04
# r_small = (0.0042, 0.00456)    
# r_medium = (0.00457, 0.00544) 
# r_large = (0.00545, 0.00580) 

#05
# r_small = (0.004, 0.00447)    
# r_medium = (0.00448, 0.00552) 
# r_large = (0.00553, 0.006) 

#06
r_small = (0.0038, 0.0044)    
r_medium = (0.00441, 0.00559) 
r_large = (0.0056, 0.0062) 

#07
# r_small = (0.0036, 0.00431)    
# r_medium = (0.00432, 0.00568) 
# r_large = (0.00569, 0.00640) 

#08
# r_small = (0.0034, 0.00423)    
# r_medium = (0.00424, 0.00576) 
# r_large = (0.00577, 0.0066) 


# Divide particles into thirds
n_per_layer = n_falling // 3
n_remainder = n_falling % 3

n_large = n_per_layer + (1 if n_remainder > 0 else 0)
n_medium = n_per_layer + (1 if n_remainder > 1 else 0)
n_small = n_per_layer

# Vertical layers (divide box into thirds)
layer_height = box_height / 3.0
margin = 0.005  # 5mm margin from walls

y_bottom = (box_bottom + margin, box_bottom + layer_height - margin)
y_middle = (box_bottom + layer_height + margin, box_bottom + 2*layer_height - margin)
y_top = (box_bottom + 2*layer_height + margin, box_top - margin)

print(f"\n{'='*60}")
print(f"Layer 1 (BOTTOM): {n_large} LARGE particles, R = {r_large[0]*1000:.1f}-{r_large[1]*1000:.1f}mm")
print(f"Layer 2 (MIDDLE): {n_medium} MEDIUM particles, R = {r_medium[0]*1000:.1f}-{r_medium[1]*1000:.1f}mm")
print(f"Layer 3 (TOP):    {n_small} SMALL particles, R = {r_small[0]*1000:.1f}-{r_small[1]*1000:.1f}mm")

def generate_position(y_min, y_max):
    # Random position within layer
    x = random.uniform(box_left + margin, box_right - margin)
    y = random.uniform(y_min, y_max)
    return [x, y, 0.0]

def generate_velocity():
    # Small random initial velocity
    return [random.uniform(-0.005, 0.003), 0.0, 0.0]

# Generate particles by layer
s_falling = []
v_falling = []
R_falling = []

# LAYER 1: Large particles at the bottom
for _ in range(n_large):
    s_falling.append(generate_position(y_bottom[0], y_bottom[1]))
    v_falling.append(generate_velocity())
    R_falling.append(random.uniform(r_large[0], r_large[1]))

# LAYER 2: Medium particles in the middle
for _ in range(n_medium):
    s_falling.append(generate_position(y_middle[0], y_middle[1]))
    v_falling.append(generate_velocity())
    R_falling.append(random.uniform(r_medium[0], r_medium[1]))

# LAYER 3: Small particles on top
for _ in range(n_small):
    s_falling.append(generate_position(y_top[0], y_top[1]))
    v_falling.append(generate_velocity())
    R_falling.append(random.uniform(r_small[0], r_small[1]))

# Convert to numpy arrays
s_falling = np.array(s_falling)
v_falling = np.array(v_falling)
R_falling = np.array(R_falling)

# Colour coordination
particletype = np.empty(n_falling, dtype=np.int8) 

for n, Rvalue in enumerate(R_falling):
    if Rvalue >= r_large[0]:
        particletype[n] = 2
    elif Rvalue >= r_medium[0]:
        particletype[n] = 1
    else:
        particletype[n] = 0

# Save particle data to csv files
WRITE("s_falling_data.csv", s_falling)
WRITE("v_falling_data.csv", v_falling)
WRITE("R_falling_data.csv", R_falling[:, np.newaxis])

np.savez(
    "falling_data.npz",
    s_falling=s_falling,
    v_falling=v_falling,
    R_falling=R_falling,
    particletype=particletype
)

print(f"  Layer heights: {layer_height*100:.1f}cm each")
print(f"{'='*60}\n")

# =======
# PARTICLE RANDOM GENERATION (commented out, as it ended up unused for our study)
# =======
"""
# Completely random positions (no layers)
def s_gen():
    return float(random.uniform(box_min+0.015, box_max-0.015))
def v_gen():
    return float(random.uniform(-0.005, 0.003))
def r_gen():
    return float(random.uniform(0.003, 0.007))

s_falling = np.array([[s_gen(), s_gen(), 0.0] for _ in range(n_falling)])
v_falling = np.array([[v_gen(), 0.0, 0.0] for _ in range(n_falling)])
R_falling = np.array([r_gen() for _ in range(n_falling)])
"""

box_info = {
    "box_left": box_left,
    "box_right": box_right,
    "box_bottom": box_bottom,
    "box_top": box_top,
    "box_width": box_width,
    "box_height": box_height,
    "layer_height": layer_height,
    "margin": margin,
    "y_bottom_min": y_bottom[0],
    "y_bottom_max": y_bottom[1],
    "y_middle_min": y_middle[0],
    "y_middle_max": y_middle[1],
    "y_top_min": y_top[0],
    "y_top_max": y_top[1]
}


WRITE_DICT("box_dimensions.csv", box_info)


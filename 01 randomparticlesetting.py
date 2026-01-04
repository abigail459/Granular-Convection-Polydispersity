# 01randomparticlesetting.py
import csv
import random
import numpy as np
import os


# --- NUMBER OF PARTICLES TO SHAKE IN CONTAINER --- 
n_falling = 264  

os.chdir("/Users/liliy/Documents/GitHub/ISS2.0/data") # Change accordingly

def WRITE(file, data):
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"\tdata written to '{file}'")

def WRITE_DICT(file, data_dict):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_dict.keys())   # Header
        writer.writerow(data_dict.values()) # Values
        print(f"\tbox info written to '{file}'")
        print("\n")


# --- BOX DIMENSIONS --- (saved in .csv file)
box_left = 0.01
box_right = 0.19
box_bottom = 0.01
box_top = 0.19

box_width = box_right -box_left   # 0.18m
box_height = box_top -box_bottom  # 0.18m

# --- TRIAL RANGES ---
# Particle size ranges (3 distinct sizes), also in the Results table in our study. 

# Example: 
# R_small = (0.003,  0.004)    # Small: 3-4mm
# R_medium = (0.0045, 0.0055) # Medium: 4.5-5.5mm
# R_large = (0.006, 0.007)    # Large: 6-7mm

#01
# R_small = (0.0048, 0.00492)    
# R_medium = (0.00593,0.00508) 
# R_large= (0.00509,0.00520) 

#02
# R_small = (0.0046, 0.00484)    
# R_medium = (0.00485, 0.00516) 
# R_large = (0.00517,0.00540) 

# #04
# R_small = (0.0042, 0.00456)    
# R_medium = (0.00457, 0.00544) 
# R_large = (0.00545,0.00580) 

#05
# R_small = (0.004, 0.00447)    
# R_medium = (0.00448,0.00552) 
# R_large = (0.00553, 0.006) 

#06
R_small = (0.0038, 0.0044)    
R_medium = (0.00441,0.00559) 
R_large = (0.0056, 0.0062) 

#07
# R_small = (0.0036, 0.00431)    
# R_medium =  (0.00432, 0.00568) 
# R_large = (0.00569, 0.00640) 

#08
# R_small =  (0.0034, 0.00423)    
# R_medium = (0.00424, 0.00576) 
# R_large= (0.00577, 0.0066) 


# Divide particles into thirds.
n_inlayer = n_falling //3
n_rem = n_falling %3 # n remainder

n_large = n_inlayer+(1 if n_rem> 0 else  0 )
n_medium = n_inlayer+ (1 if n_rem >1 else 0)
n_small =n_inlayer

# Vertical layers (divide box into thirds)
layer_h = box_height/3.0
margin = 0.005  # 5mm margin from walls

y_bottom = ((box_bottom +margin), (box_bottom + layer_h - margin))
y_middle = ((box_bottom + layer_h +margin), (box_bottom + 2*layer_h - margin))
y_top = ((box_bottom + 2*layer_h + margin), (box_top - margin))

print(f"\nLayer 1 (Bottom): L particles, R = {R_large[0] }-{R_large[1]}m")
print(f"Layer 2 (Middle): M particles, R = {R_medium[0]}-{R_medium[1] }m")
print(f"Layer 3 (Top):    S particles, R = {R_small[0]}-{R_small[1]}m")

def gen_pos(y_min, y_max):
    x = random.uniform(box_left + margin, box_right - margin)
    y = random.uniform(y_min, y_max)
    return [x, y, 0.0]

def gen_v():
    return [random.uniform(-0.005, 0.003), 0.0, 0.0]

# Generate particles by layer.
s_falling =[]
v_falling = []
R_falling = []

# --- LAYER 1 ---
for _ in range(n_large):
    s_falling.append(gen_pos(y_bottom[0], y_bottom[1]))
    v_falling.append(gen_v())
    R_falling.append(random.uniform(R_large[0], R_large[1]))

# --- LAYER 2 ---
for _ in range(n_medium) :
    s_falling.append(gen_pos(y_middle[0], y_middle[1]))
    v_falling.append(gen_v())
    R_falling.append(random.uniform(R_medium[0], R_medium[1]))

# --- LAYER 3 ---
for _ in range( n_small):
    s_falling.append(gen_pos (y_top[0], y_top [1]))
    v_falling.append(gen_v())
    R_falling.append( random.uniform(R_small[0], R_small[1]))

# Convert to numpy arrays
s_falling = np.array(s_falling)
v_falling =np.array(v_falling)
R_falling = np.array(R_falling)

# Colour 
particle_type = np.empty(n_falling, dtype=np.int8) 
for n, R_val in  enumerate(R_falling): # R value
    if R_val >= R_large[0]: 
        particle_type[n] = 2
    elif R_val>= R_medium[0]:
        particle_type[n] = 1
    else:
        particle_type[n]= 0

# Save particle data to csv files. 
WRITE("s_falling_data.csv",  s_falling)
WRITE("v_falling_data.csv", v_falling)
WRITE("R_falling_data.csv", R_falling[:, np.newaxis])

np.savez(
    "falling_data.npz",
    s_falling=s_falling,
    v_falling=v_falling,
    R_falling=R_falling,
    particle_type=particle_type
)

print(f"Layer heights: {layer_h*100:.1f}cm each")

# --- PARTICLE RANDOM GENERATION (commented out, as it ended up unused for our study) ---
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

# Cast into a dictionary which is then stored into a csv file
box_info = {
    "box_left": box_left,
    "box_right": box_right,
    "box_bottom": box_bottom,
    "box_top": box_top,
    "box_width": box_width,
    "box_height": box_height,
    "layer_h": layer_h,
    "margin": margin,
    "y_bottom_min": y_bottom [0],
    "y_bottom_max": y_bottom[1],
    "y_middle_min": y_middle[0],
    "y_middle_max": y_middle[1],
    "y_top_min": y_top[0] ,
    "y_top_max": y_top[1]
}


WRITE_DICT( "box_dimensions.csv", box_info)


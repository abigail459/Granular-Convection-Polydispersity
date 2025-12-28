# 03plottinggraphs.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import csv
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import time as time_module



### DIRECTORY SETUP
rootdir = "/Users/liliy/Documents/GitHub"
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()

### DATA SETUP
data = np.load("generated_values.npz")

n_frames = data["n_frames"]
s_history = data["s_history"]
R = data["R"]
n_falling = int(data["n_falling"])
time_history = data["time_history"]
rdata = np.load("falling_data.npz")
particletype = rdata["particletype"]

'''
def compute_seg_index_over_time(s_history, R, n_falling):
    """
    Compute segregation index S(t) for each saved frame.
    S(t) = N_large,top / N_large
    where 'large' is defined as the top third of radii among falling particles,
    and 'top' means centres in the top 25% of the instantaneous bed height.
    """
    # Only falling particles are considered for size classification
    all_R = R[:n_falling]
    R_min = np.min(all_R)
    R_max = np.max(all_R)

    # same idea as your colour cutoffs: top third = large
    highcutoff = R_max - (R_max - R_min) / 3.0

    is_large = all_R > highcutoff
    large_indices = np.where(is_large)[0]
    N_large = len(large_indices)

    n_frames = s_history.shape[0]
    S_values = np.zeros(n_frames)

    for k in range(n_frames):
        # positions of falling particles in this frame
        s_fall = s_history[k, :n_falling, :]
        y_coords = s_fall[:, 1]

        # define instantaneous bed height from falling particles
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        bed_height = y_max - y_min

        if bed_height <= 0.0 or N_large == 0:
            S_values[k] = 0.0
            continue

        # top 25% of bed
        top_threshold = y_min + 0.75 * bed_height

        # y of large particles only
        large_y = y_coords[large_indices]
        N_large_top = np.sum(large_y >= top_threshold)

        S_values[k] = N_large_top / N_large

    return S_values

# compute S(t) once
S_values = compute_seg_index_over_time(s_history, R, n_falling)
np.save("S_values.npy", S_values)   # optional: save for later analysis
'''

osc_enable_x = bool(data.get("oscillation_enable_x", False))
osc_enable_y = bool(data.get("oscillation_enable_y", True))

amp_x = float(data.get("oscillation_amplitude_x", 0.003))
amp_y = float(data.get("oscillation_amplitude_y", 0.003))
freq_x = float(data.get("oscillation_frequency_x", 2.0))
freq_y = float(data.get("oscillation_frequency_y", 2.0))

# plot_min = data["plot_min"]
# plot_max = data["plot_max"]


### SIMULATION PARAMETERS
t_step = float(data.get("t_step"))  # timestep
simulation_duration = float(data.get("simulation_duration"))  
display_fps = float(data.get("display_fps")) #fps
numrendered = display_fps*simulation_duration
    
## FUNCTIONS
if osc_enable_y and not osc_enable_x:
    oscillation_amplitude = amp_y
    oscillation_frequency = freq_y
elif osc_enable_x and not osc_enable_y:
    oscillation_amplitude = amp_x
    oscillation_frequency = freq_x
else:
    # if both enabled then show magnitude of vector (?) right....
    oscillation_amplitude = max(amp_x, amp_y)
    oscillation_frequency = max(freq_x, freq_y)

omega = 2 * np.pi * oscillation_frequency

def get_box_displacement(time):
    return oscillation_amplitude * np.sin(omega * time)

def get_box_velocity(time):
    return oscillation_amplitude * omega * np.cos(omega * time)

### INITIALISATION
def initial_render(R, n_falling): # render constant variables first, then update movement using FuncAnimation
    fig = plt.figure(figsize=(6, 6), dpi=80)  # Smaller/lower DPI = faster

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    # Fixed limits --> full screen
    ax.set_xlim(0, 0.2) # 0.2
    ax.set_ylim(0, 0.2) # 0.2
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    

    highcutoff = np.max(R)-((np.max(R)-np.min(R))/3) # relative to size sample, computed once for later use
    lowcutoff = np.min(R)+((np.max(R)-np.min(R))/3)

    circles = []
    texts = []

    # falling particles (coloured)    
    for i in range(n_falling):
        circle = Circle((0, 0), R[i], edgecolor='black', alpha=0.9, linewidth=2)
        ax.add_patch(circle)
        circles.append(circle)
        
        text = ax.text(0, 0, str(i+1), ha='center', va='center', fontsize=9, 
                fontweight='bold', color='white')
        texts.append(text)

    # box walls (gray circles)
    for i in range(n_falling, len(R)):
        # x, y = s_current[i, 0], s_current[i, 1]
        circle = Circle((0, 0), R[i], edgecolor='none', facecolor='#202020', alpha=1.0) # not actual animation; just setting up
        ax.add_patch(circle)
        circles.append(circle)
        texts.append(None)


    title = ax.set_title("", fontsize=12, fontweight='bold', pad=10)

    # Light grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5) 
    
    return fig, ax, circles, texts, title, highcutoff, lowcutoff


### UPDATE FRAME
def update_frame(frame, s_history, times, R, circles, texts, title, n_falling, highcutoff, lowcutoff):
    s_current = s_history[frame]
    time = times[frame]

    for n, circle in enumerate(circles):
        x, y = s_current[n, 0], s_current[n, 1]        
        circle.center = (x, y)

        if n < n_falling: # setting colours
            if particletype[n] == 2:
                circle.set_facecolor("#004CFF")
            elif particletype[n] == 0:
                circle.set_facecolor("#FF0000")
            else:
                circle.set_facecolor("#33FF33")

            # if n == 137:
            #     circle.set_facecolor("#004CFF")
            # else:
            #     circle.set_facecolor("#FF0C00")
            if texts[n] is not None:
                texts[n].set_position((x, y))

    # title
    box_disp = get_box_displacement(time)
    title.set_text(f'Time: {time:.2f}s | Oscillation: {box_disp*1000:.1f}mm')

    return circles + [text for text in texts if text is not None] + [title] # for blitting (faster way to copy images)


### RUNNING ANIMATION
fig, ax, circles, texts, title, highcutoff, lowcutoff = initial_render(R, n_falling)
start_time = time_module.time()

animation = FuncAnimation(fig =fig, 
                          func = update_frame, 
                          frames = len(s_history), 
                          fargs = (s_history, time_history, R, circles, texts, title, n_falling, highcutoff, lowcutoff), # args for update_frame
                          blit = True, 
                          interval=20 # delay between consecutive frames of an animation
                          )


### MAKING FRAMES
def render_frame(frame_index, filename=None):
    s_current = s_history[frame_index]
    time = time_history[frame_index]
    box_disp = get_box_displacement(time)

    for n, circle in enumerate(circles):
        x, y = s_current[n, 0], s_current[n, 1]        
        circle.center = (x, y)

        if n < n_falling: # setting colours
            if particletype[n] == 2:
                circle.set_facecolor("#004CFF")
            elif particletype[n] == 0:
                circle.set_facecolor("#FF0000")
            else:
                circle.set_facecolor("#33FF33")
            if texts[n] is not None:
                texts[n].set_position((x, y))

    # title
    box_disp = get_box_displacement(time)
    title.set_text(f'Time: {time:.2f}s | Oscillation: {box_disp*1000:.1f}mm')

    fig.canvas.draw() # draw_idle if interactive
    if filename:
        fig.savefig(filename, dpi=80)

# render all
render_frames = False
if render_frames:
    os.chdir(f"{rootdir}/ISS2.0/Figures/")
    os.makedirs("Frames", exist_ok=True) # exist_ok prevents errors if file is alr there
    for frame in range(len(s_history)):
        render_frame(frame, filename=f"Frames/fig_{frame:04d}.png")
    print(f"Rendered frames")


### END OUTPUT + SAVING ANIMATION
# total_time = time_module.time() - start_time
os.chdir(f"{rootdir}/ISS2.0/Figures")
# name = f"output-f{freq_y}-a{amp_y}"

animation.save("output.mp4", fps=display_fps, dpi=80)
print("\n"+"-"*60)
print(f"Saved video as 'output.mp4' with freq={freq_y} and amp={amp_y}")
print("-"*60)



'''
def plot_S_vs_time(time_history, S_values):
    plt.figure()
    plt.plot(time_history, S_values, '-k')
    plt.xlabel("Time (s)")
    plt.ylabel("Segregation index S")
    plt.tight_layout()
    plt.savefig("S_vs_time.png", dpi=150)
    plt.close()

plot_S_vs_time(time_history, S_values)
'''

# 03plottinggraphs.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle
import csv
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import time as time_module
from matplotlib.animation import FFMpegWriter


# == DIRECTORY SETUP ==
rootdir = "/Users/Abigail/Documents/GitHub" # Change accordingly
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()

# == DATA SETUP == 
data = np.load("generated_values.npz")

n_frames = data["n_frames"]
s_history = data["s_history"]
R = data["R"]
n_falling = int(data["n_falling"])
time_history = data["time_history"]
rdata = np.load("falling_data.npz")
particletype = rdata["particletype"]


osc_enable_x = bool(data.get("oscillation_enable_x", False))
osc_enable_y = bool(data.get("oscillation_enable_y", True))

amp_x = float(data.get("oscillation_amplitude_x", 0.003))
amp_y = float(data.get("oscillation_amplitude_y", 0.003))
freq_x = float(data.get("oscillation_frequency_x", 2.0))
freq_y = float(data.get("oscillation_frequency_y", 2.0))


# == SIMULATION PARAMETERS == 
t_step = float(data.get("t_step"))  # Timestep
simulation_duration = float(data.get("simulation_duration"))  
display_fps = float(data.get("display_fps")) # FPS
numrendered = display_fps*simulation_duration
    
# == FUNCTIONS == 
if osc_enable_y and not osc_enable_x:
    oscillation_amplitude = amp_y
    oscillation_frequency = freq_y
elif osc_enable_x and not osc_enable_y:
    oscillation_amplitude = amp_x
    oscillation_frequency = freq_x
else:
    oscillation_amplitude = max(amp_x, amp_y)
    oscillation_frequency = max(freq_x, freq_y)

omega = 2 * np.pi * oscillation_frequency

def get_box_displacement(time):
    return oscillation_amplitude * np.sin(omega * time)

def get_box_velocity(time):
    return oscillation_amplitude * omega * np.cos(omega * time)

# == INITIALISATION == 
def initial_render(R, n_falling):   # Render constant variables first, then update movement using FuncAnimation
    fig = plt.figure(figsize=(6, 6), dpi=80) 

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    # Fixed limits: full screen
    ax.set_xlim(0, 0.2) # 0.2
    ax.set_ylim(0, 0.2) # 0.2
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    

    highcutoff = np.max(R)-((np.max(R)-np.min(R))/3) # Relative to size sample, computed once for later use
    lowcutoff = np.min(R)+((np.max(R)-np.min(R))/3)

    circles = []
    texts = []

    # Falling particles (coloured)    
    for i in range(n_falling):
        circle = Circle((0, 0), R[i], edgecolor='black', alpha=0.9, linewidth=2)
        ax.add_patch(circle)
        circles.append(circle)
        
        text = ax.text(0, 0, str(i+1), ha='center', va='center', fontsize=9, 
                fontweight='bold', color='white')
        texts.append(text)

    # Box walls (gray circles)
    for i in range(n_falling, len(R)):
        circle = Circle((0, 0), R[i], edgecolor='none', facecolor='#202020', alpha=1.0) # Not actual animation; just setting up
        ax.add_patch(circle)
        circles.append(circle)
        texts.append(None)


    title = ax.set_title("", fontsize=12, fontweight='bold', pad=10)

    # Light grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5) 
    
    return fig, ax, circles, texts, title, highcutoff, lowcutoff


# == UPDATE FRAME ==
def update_frame(frame, s_history, times, R, circles, texts, title, n_falling, highcutoff, lowcutoff):
    s_current = s_history[frame]
    time = times[frame]

    for n, circle in enumerate(circles):
        x, y = s_current[n, 0], s_current[n, 1]        
        circle.center = (x, y)

        if n < n_falling: # Setting particle colours
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

    return circles + [text for text in texts if text is not None] + [title] # For blitting, a faster way to copy images


# == RUNNING ANIMATION ==
fig, ax, circles, texts, title, highcutoff, lowcutoff = initial_render(R, n_falling)
start_time = time_module.time()

print(texts)
animation = FuncAnimation(fig =fig, 
                          func = update_frame, 
                          frames = len(s_history), 
                          fargs = (s_history, time_history, R, circles, texts, title, n_falling, highcutoff, lowcutoff), # Arguments for update_frame
                          blit = True, 
                          interval=20 # Delay between consecutive frames of an animation
                          )


### MAKING FRAMES
def render_frame(frame_index, filename=None):
    s_current = s_history[frame_index]
    time = time_history[frame_index]
    box_disp = get_box_displacement(time)

    for n, circle in enumerate(circles):
        x, y = s_current[n, 0], s_current[n, 1]        
        circle.center = (x, y)

        if n < n_falling: # Setting colours
            if particletype[n] == 2:
                circle.set_facecolor("#004CFF")
            elif particletype[n] == 0:
                circle.set_facecolor("#FF0000")
            else:
                circle.set_facecolor("#33FF33")
            if texts[n] is not None:
                texts[n].set_position((x, y))

    # Title
    box_disp = get_box_displacement(time)
    title.set_text(f'Time: {time:.2f}s | Oscillation: {box_disp*1000:.1f}mm')

    fig.canvas.draw()
    if filename:
        fig.savefig(filename, dpi=80)

# Render all
render_frames = False
if render_frames:
    os.chdir(f"{rootdir}/ISS2.0/Figures/")
    os.makedirs("Frames", exist_ok=True) # exist_ok prevents errors if file is already there
    for frame in range(len(s_history)):
        render_frame(frame, filename=f"Frames/fig_{frame:04d}.png")
    print(f"Rendered frames")


### END OUTPUT + SAVING ANIMATION
os.chdir(f"{rootdir}/ISS2.0/Figures")

print("\nStarting video compilation...")
total_frames = len(s_history)

writer = FFMpegWriter(
    fps=display_fps,
    metadata=dict(artist="ISS2.0"),
    bitrate=1800
)

with writer.saving(fig, "output.mp4", dpi=80):
    for i in range(total_frames):
        update_frame(
            i, s_history, time_history, R,
            circles, texts, title,
            n_falling, highcutoff, lowcutoff
        )
        writer.grab_frame()

        # Progress update every 5%
        if i % max(1, total_frames // 20) == 0:
            percent = 100 * i / total_frames
            elapsed = time_module.time() - start_time
            print(f"  {percent:5.1f}% complete | {i}/{total_frames} frames | {elapsed:.1f}s elapsed")

print("\n" + "-"*60)
print("Video compilation complete!")
print(f"Saved video as 'output.mp4'")
print("-"*60)


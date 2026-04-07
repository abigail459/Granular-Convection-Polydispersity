import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np



data = np.load("data/generated_values.npz")
s_history = data["s_history"]
v_history = data["v_history"]
R = data["R"]
print(len(s_history))

fig = plt.figure(figsize=(6, 6), dpi=80) 
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlim(0, 0.2) # 0.2
ax.set_ylim(0, 0.2) # 0.2
ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')

colours = ["#FF0000", "#EE8F00", "#FFEE00", "#00FF00", "#0000FF", "#FF00FF"]

# bleh = plt.arrow(0.01, 0.01, -0.01, 0.01)
for timestep in range(0, len(s_history), 20):
    track = 88
    particle = s_history[timestep][track]
    vel = v_history[timestep][track]
    i = len(s_history)/7
    colour = 0
    if timestep < i:
        colour = 0
    elif timestep < i*2:
        colour = 1
    elif timestep < i*3:
        colour = 2
    elif timestep < i*4:
        colour = 3
    elif timestep < i*5:
        colour = 4
    else:
        colour = 5
    ax.add_patch(Circle((particle[0], particle[1]), R[track], alpha=0.1, color="#000000"))
    plt.arrow(particle[0], particle[1], vel[0]/100, vel[1]/100, alpha=0.45, color=colours[colour])
    
    
# for timestep in s_history[0]:
#     plt.arrow(timestep[0], timestep[1], 0.005, 0.005, alpha=0.5)
#     # ax.add_patch(bleh)
# for i in s_history[1]:
#     plt.arrow(i[0], i[1], 0.005, 0.005, alpha=0.5, color="#FF0000")
#     # ax.add_patch(bleh)
plt.show()
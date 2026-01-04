import numpy as np
import os
import matplotlib.pyplot as plt
import csv

print("--- Segregation analysis started ---")

# --- PATH ---
os.chdir("/Users/Abigail/Documents/GitHub/ISS2.0/data")


# --- LOAD DATA ---
print("Loading simulation data...")
data = np.load("generated_values.npz", allow_pickle=True)

s_hist = data["s_hist"]          # (frames, N, 3)
R = data["R"]
n_falling = int(data["n_falling"])
time = np.array(data["time_history"])
fps = float(data["display_fps"])

n_frames = len(s_hist)

print(f"Loaded {n_frames} frames")
print(f"Falling particles: {n_falling}")
print(f"Time range: {time[0]:.2f}s to {time[-1]:.2f}s")
print(f"fps used for smoothing: {fps}")


# --- LARGE PARTICLES ---
print("Identifying large particles...")

R_falling = R [:n_falling]
R_min = np.min(R_falling)
R_max = np.max(R_falling)

large_bed = R_max - (R_max - R_min) /3.0  #  particles that are in the top 33% of the particles (large particles)
large_indices = np.where(R_falling >= large_bed)[0]
N_large = len(large_indices)

print(f"Large particle cutoff radius: {large_bed:.4e} m")
print(f"Number of large particles: {N_large}")


# --- SEGREGATION INDEX S(t) ---
def segregation_index(positions):
    """
    S(t) = fraction of large particles whose centres
    lie in the top 25% of the instantaneous bed height
    """
    if N_large == 0:
        return 0.0

    y_all = positions[:n_falling, 1]
    y_large = positions[large_indices, 1]

    bed_bottom = np.min(y_all)
    bed_top = np.max(y_all)
    bed_height = bed_top - bed_bottom

    if bed_height <= 0:
        return 0.0

    top_region = bed_top - 0.25 * bed_height
    return np.sum(y_large >= top_region) / N_large

# --- FIND S(t) ---
print("Computing segregation index S(t)")

S = np.zeros(n_frames)
for i in range(n_frames):
    if i % 500 == 0:
        print(f"  Processing frame {i}/{n_frames}")
    S[i] = segregation_index(s_hist[i])

print("Segregation index computation complete")


# --- SMOOTHING (1s moving average) ---
print("Smoothing S(t)")

windw = max(1, int(1.0 * fps))
kernel = np.ones(windw) / windw

S_smth = np.convolve(S, kernel, mode= "valid") #by smth , it means "smooth". 
t_smth = time[:len(S_smth)]

print(f"Smoothing windw: {windw} frames (~1 s).")


# --- METRICS ---
print("Extracting segregation metrics")

S_max = float(np.max(S_smth))

Smax_threshold = 0.9 * S_max
t_90 = float(t_smth[np.argmax(S_smth >= Smax_threshold)]) # finds index of highest value .. 

t_peak = t_smth[np.argmax(S_smth)]

plateau_mask = np.abs(t_smth - t_peak) <= 10.0

S_plateau = float(np.mean(S_smth[plateau_mask]))
S_plateau_std = float(np.std(S_smth[plateau_mask]))

print(f"S_max      = {S_max:.4f}")
print(f"t_90       = {t_90:.3f}s")
print(f"S_plateau  = {S_plateau:.4f} ± {S_plateau_std:.4f}")


# --- PLOT ---
print("Saving segregation_vs_time.png")

plt.figure(figsize=(7, 4))
plt.plot(time, S, color="gray", alpha=0.4, label="Raw S(t)")
plt.plot(t_smth, S_smth, color="blue", lw=2, label="Smoothed S(t)")

plt.axhline(S_max, color="red", ls="--", label=r"$S_{\max}$")
plt.axvline(t_90, color="green", ls="--", label=r"$t_{90}$")

plt.xlabel("Time(s)")
plt.ylabel("Segregation index, S")
plt.legend()
plt.tight_layout()

plt.savefig("segregation_vs_time.png", dpi=300)
plt.close()
print("Plot saved")

# --- WRITE CSV (append) ---
print("Writing results to segregation_results.csv")

csv_file = "segregation_results.csv"

header = ["S_plateau","S_plateau_std", "S_max","t_90_s"]

write_header = True
if os.path.exists(csv_file):
    with open(csv_file, "r") as f:
        first_line = f.readline()
        if "S_plateau" in first_line:
            write_header = False

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)

    writer.writerow([S_plateau, S_plateau_std, S_max, t_90])

print("Results written successfully.")
print("--- Analysis completed successfully!! ---")

import numpy as np
import os
import matplotlib.pyplot as plt
import csv

print("=== Segregation analysis started ===")

# === PATHS ===
ROOT = "/Users/liliy/Documents/GitHub/ISS2.0"
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = DATA_DIR  # Keep outputs with data

os.chdir(DATA_DIR)


# === LOAD DATA ====
print("Loading simulation data...")
data = np.load("generated_values.npz", allow_pickle=True)

s_history = data["s_history"]          # (frames, N, 3)
R = data["R"]
n_falling = int(data["n_falling"])
time = np.array(data["time_history"])
fps = float(data["display_fps"])

n_frames = len(s_history)

print(f"Loaded {n_frames} frames")
print(f"Falling particles: {n_falling}")
print(f"Time range: {time[0]:.2f} s → {time[-1]:.2f} s")
print(f"FPS used for smoothing: {fps}")


# === IDENTIFY LARGE PARTICLES ===
print("Identifying large particles...")

R_fall = R[:n_falling]
R_min = np.min(R_fall)
R_max = np.max(R_fall)

large_cutoff = R_max - (R_max - R_min) / 3.0
large_indices = np.where(R_fall >= large_cutoff)[0]
N_large = len(large_indices)

print(f"Large particle cutoff radius: {large_cutoff:.4e} m")
print(f"Number of large particles: {N_large}")


# === SEGREGATION INDEX FUNCTION ===
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

# === COMPUTE S(t) ===
print("Computing segregation index S(t)")

S = np.zeros(n_frames)

for i in range(n_frames):
    if i % 500 == 0:
        print(f"  Processing frame {i}/{n_frames}")
    S[i] = segregation_index(s_history[i])

print("Segregation index computation complete")


# --- SMOOTHING (1s moving average) ---
print("Smoothing S(t)")

window = max(1, int(1.0 * fps))
kernel = np.ones(window) / window

S_smooth = np.convolve(S, kernel, mode="valid")
time_smooth = time[:len(S_smooth)]

print(f"Smoothing window: {window} frames (~1 s).")


# --- METRICS ---
print("Extracting segregation metrics")

S_max = float(np.max(S_smooth))

threshold = 0.9 * S_max
idx_90 = np.argmax(S_smooth >= threshold)
t_90 = float(time_smooth[idx_90])

idx_peak = np.argmax(S_smooth)
t_peak = time_smooth[idx_peak]

plateau_mask = np.abs(time_smooth - t_peak) <= 10.0

S_plateau = float(np.mean(S_smooth[plateau_mask]))
S_plateau_std = float(np.std(S_smooth[plateau_mask]))

print(f"S_max      = {S_max:.4f}")
print(f"t_90       = {t_90:.3f} s")
print(f"S_plateau  = {S_plateau:.4f} ± {S_plateau_std:.4f}")


# === PLOT ===
print("Saving segregation_vs_time.png")

plt.figure(figsize=(7, 4))
plt.plot(time, S, color="gray", alpha=0.4, label="Raw S(t)")
plt.plot(time_smooth, S_smooth, color="blue", lw=2, label="Smoothed S(t)")

plt.axhline(S_max, color="red", ls="--", label=r"$S_{\max}$")
plt.axvline(t_90, color="green", ls="--", label=r"$t_{90}$")

plt.xlabel("Time (s)")
plt.ylabel("Segregation index S")
plt.legend()
plt.tight_layout()
plt.savefig("segregation_vs_time.png", dpi=300)
plt.close()

print("Plot saved")

# === WRITE CSV (append) ===
print("Writing results to segregation_results.csv")

csv_file = "segregation_results.csv"

header = [
    "S_plateau",
    "S_plateau_std",
    "S_max",
    "t_90_s"
]

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

    writer.writerow([
        S_plateau,
        S_plateau_std,
        S_max,
        t_90
    ])

print("Results written successfully")
print("=== Analysis completed successfully!! ===")

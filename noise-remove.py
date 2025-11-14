import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, welch
import os
import glob

# ----------------------------
# 1. Define Paths and Parameters
# ----------------------------
# INPUT_FOLDER = './EMG-dataset/TrainCSV_C23/TrainCSV_C23/' # Look for CSVs in the current directory
# OUTPUT_DATA_FOLDER = 'data/train/'
# OUTPUT_PLOT_FOLDER = 'plots/train/'
INPUT_FOLDER = './EMG-dataset/TestCSV_C23/TestCSV_C23/'
OUTPUT_DATA_FOLDER = 'data/test/'
OUTPUT_PLOT_FOLDER = 'plots/test/'

# Filter parameters from the research paper
FS = 2000.0           # Sampling frequency (Hz)
LOW_PASS_CUTOFF = 500.0 # Low-pass cutoff frequency (Hz)
NOTCH_FREQ = 50.0     # Notch filter frequency (Hz)
FILTER_ORDER = 1      # Filter order (paper specifies "first-order")

# ----------------------------
# 2. Create Output Directories
# ----------------------------
os.makedirs(OUTPUT_DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_PLOT_FOLDER, exist_ok=True)
print(f"Ensured directories exist:\n- {OUTPUT_DATA_FOLDER}\n- {OUTPUT_PLOT_FOLDER}")

# ----------------------------
# 3. Design Filters (do this once)
# ----------------------------
nyquist = 0.5 * FS
# Low-pass filter design (Butterworth)
b_low, a_low = butter(FILTER_ORDER, LOW_PASS_CUTOFF / nyquist, btype="low")

# Notch filter design (IIR)
notch_quality = 30.0
b_notch, a_notch = iirnotch(w0=NOTCH_FREQ / nyquist, Q=notch_quality)
print("Filters designed.")

# ----------------------------
# 4. Find and Process All CSV Files
# ----------------------------
# Find all files ending in .csv in the input folder
csv_files = glob.glob(os.path.join(INPUT_FOLDER, '*.csv'))
print(f"Found {len(csv_files)} CSV file(s) to process: {csv_files}")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # --- Skip already filtered files to avoid re-processing ---
    if base_name.endswith('_filtered'):
        print(f"\n--- Skipping already filtered file: {file_name} ---")
        continue
        
    print(f"\n--- Processing: {file_name} ---")

    # Load data
    df = pd.read_csv(file_path)
    
    # Identify channels
    channels = [col for col in df.columns if col.startswith("Chan_")]
    if not channels:
        print(f"No 'Chan_' columns found in {file_name}. Skipping.")
        continue
    
    print(f"Found channels: {channels}")
    
    # ----------------------------
    # 5. Apply Filters
    # ----------------------------
    filtered_df = df.copy()
    for chan in channels:
        signal_data = df[chan].values
        # Apply 50Hz notch filter first (as per paper)
        signal_notched = filtfilt(b_notch, a_notch, signal_data)
        # Apply 500Hz low-pass filter second
        filtered_signal = filtfilt(b_low, a_low, signal_notched)
        filtered_df[chan] = filtered_signal
    
    # ----------------------------
    # 6. Save Filtered Data
    # ----------------------------
    output_data_path = os.path.join(OUTPUT_DATA_FOLDER, f"{base_name}_filtered.csv")
    filtered_df.to_csv(output_data_path, index=False)
    print(f"Saved filtered data to: {output_data_path}")

    # ----------------------------
    # 7. Generate Multi-Channel Plot
    # ----------------------------
    num_channels = len(channels)
    # Create a plot with N rows (one for each channel) and 2 columns (Time, PSD)
    fig, axes = plt.subplots(nrows=num_channels, ncols=2, figsize=(18, 5 * num_channels), squeeze=False)
    
    t = np.arange(len(df)) / FS
    two_sec = int(2 * FS) # Number of samples for 2 seconds
    
    # Ensure we don't try to plot more than we have
    if two_sec > len(t):
        two_sec = len(t)
        
    for i, chan in enumerate(channels):
        # --- Time Domain Plot (Left Column) ---
        ax_time = axes[i, 0]
        ax_time.plot(t[:two_sec], df[chan].values[:two_sec], label="Original", alpha=0.7)
        ax_time.plot(t[:two_sec], filtered_df[chan].values[:two_sec], label="Filtered", alpha=0.9)
        ax_time.set_title(f"{chan} - Time Domain (First 2 sec)")
        ax_time.set_ylabel("Amplitude")
        ax_time.legend()
        ax_time.grid(True)
        
        # --- Frequency Domain Plot (Right Column) ---
        ax_psd = axes[i, 1]
        f_orig, Pxx_orig = welch(df[chan].values, FS, nperseg=1024)
        f_filt, Pxx_filt = welch(filtered_df[chan].values, FS, nperseg=1024)
        
        ax_psd.semilogy(f_orig, Pxx_orig, label="Original PSD", alpha=0.7)
        ax_psd.semilogy(f_filt, Pxx_filt, label="Filtered PSD", alpha=0.9)
        
        ax_psd.axvline(LOW_PASS_CUTOFF, color="r", linestyle="--", label="Low-pass (500Hz)")
        ax_psd.axvline(NOTCH_FREQ, color="m", linestyle="--", label="Notch (50Hz)")
        
        ax_psd.set_title(f"{chan} - Power Spectral Density (PSD)")
        ax_psd.set_ylabel("Power/Frequency (dB/Hz)")
        ax_psd.set_xlim(0, FS / 2)
        ax_psd.legend()
        ax_psd.grid(which='both', linestyle='--', alpha=0.6)

    # Set x-axis labels only on the bottom-most plots
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    
    # Add a main title for the whole figure
    fig.suptitle(f"Filtering Comparison for {file_name}", fontsize=20, y=1.03)
    
    # Adjust layout and save
    plt.tight_layout()
    output_plot_path = os.path.join(OUTPUT_PLOT_FOLDER, f"{base_name}_comparison_plot.png")
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to save memory
    
    print(f"Saved multi-channel plot to: {output_plot_path}")

print("\n--- All files processed. ---")
import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from numpy.linalg import lstsq

# Parameters (EMAHA-DB1)
FS = 2000.0             # Sampling frequency (Hz)
WIN_SEC = 0.25          # Window size: 250 ms (per paper)
WIN_LEN = int(WIN_SEC * FS)  # 250 ms × 2000 Hz = 500 samples
OVERLAP = 0.5           # 50% overlap
STEP = int(WIN_LEN * (1 - OVERLAP))  # step size = 250 samples
THR_FACTOR = 0.01       # Threshold for ZC/SSC/WAMP

DATA_DIR = "data/train"            
OUTPUT_DIR = "data/processed/train"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🔧 Feature extraction config:")
print(f"   - Sampling Rate: {FS} Hz")
print(f"   - Window Length: {WIN_LEN} samples ({WIN_SEC*1000:.0f} ms)")
print(f"   - Overlap: {OVERLAP*100:.0f}% → Step size: {STEP} samples\n")


def remove_baseline(x, window_size=1000):
    """Removing slow DC drift using moving average baseline subtraction."""
    if len(x) < window_size:
        window_size = max(3, len(x) // 2)
    kernel = np.ones(window_size) / window_size
    baseline = np.convolve(x, kernel, mode='same')
    return x - baseline

def channel_standardize(x):
    """Normalize per-segment to zero mean and unit variance."""
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


# Basic feature functions (F0–F1)
def mean_absolute_value(x): return np.mean(np.abs(x))
def root_mean_square(x): return np.sqrt(np.mean(x**2))
def variance(x): return np.var(x)
def waveform_length(x): return np.sum(np.abs(np.diff(x)))
def simple_square_integral(x): return np.sum(x**2)
def difference_absolute_standard_deviation_value(x): return np.std(np.diff(np.abs(x)))
def difference_absolute_mean_value(x): return np.mean(np.abs(np.diff(x)))

def zero_crossings(x, thr):
    s = np.sign(x)
    return np.sum((s[:-1] * s[1:] < 0) &
                  ((np.abs(x[:-1]) > thr) | (np.abs(x[1:]) > thr)))

def slope_sign_changes(x, thr):
    ssc = 0
    for i in range(1, len(x) - 1):
        if ((x[i] - x[i-1]) * (x[i] - x[i+1])) > thr:
            ssc += 1
    return ssc

def willison_amplitude(x, thr):
    return np.sum(np.abs(np.diff(x)) > thr)

def median_frequency(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    c = np.cumsum(Pxx)
    return f[np.searchsorted(c, c[-1] / 2)]

def mean_frequency(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    return np.sum(f * Pxx) / (np.sum(Pxx) + 1e-12)

def log_mean_frequency(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    return np.exp(np.sum(np.log(f + 1e-12) * Pxx) / (np.sum(Pxx) + 1e-12))

def power_spectral_density(x, fs):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    return f, Pxx

# Advanced feature (F2–F9)
def autoregressive_coeffs(x, order=4):
    N = len(x)
    if N <= order:
        return np.zeros(order)
    X = np.column_stack([x[i:N - order + i] for i in range(order)])
    y = x[order:]
    a, _, _, _ = lstsq(X, y, rcond=None)
    return a

def spectral_band_powers(x, fs):
    f, Pxx = power_spectral_density(x, fs)
    bands = [(20, 150), (150, 300), (300, 500)]
    sbp = []
    for low, high in bands:
        mask = (f >= low) & (f < high)
        sbp.append(np.trapezoid(Pxx[mask], f[mask])) 
    return sbp

def local_binary_pattern_feature(x):
    x2 = np.sign(np.diff(x))
    return np.sum(x2 > 0) / (len(x2) + 1e-12)

def channel_correlation(ch_data):
    corrs = []
    for i in range(len(ch_data)):
        for j in range(i + 1, len(ch_data)):
            corr = np.corrcoef(ch_data[i], ch_data[j])[0, 1]
            corrs.append(corr)
    return np.mean(corrs) if corrs else 0.0

def temporal_spatial_descriptors(ch_data):
    arr = np.array(ch_data)
    means = np.mean(arr, axis=1)
    stds = np.std(arr, axis=1)
    corr = channel_correlation(ch_data)
    return np.concatenate([means, stds, [corr]])

def normalized_temporal_spatial_descriptors(ch_data):
    tsd = temporal_spatial_descriptors(ch_data)
    return (tsd - np.min(tsd)) / (np.ptp(tsd) + 1e-12)

def invariant_time_domain_descriptor(x):
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return [mean_absolute_value(x_norm),
            waveform_length(x_norm),
            root_mean_square(x_norm)]


# Extracting features for one channel (F0–F9)
def extract_features(x, fs, thr):
    feats = {}

    # F0: Time-domain
    feats["MAV"] = mean_absolute_value(x)
    feats["RMS"] = root_mean_square(x)
    feats["VAR"] = variance(x)
    feats["WL"] = waveform_length(x)
    feats["ZC"] = zero_crossings(x, thr)
    feats["SSC"] = slope_sign_changes(x, thr)
    feats["WAMP"] = willison_amplitude(x, thr)
    feats["SSI"] = simple_square_integral(x)

    # F1: Frequency-domain
    feats["MDF"] = median_frequency(x, fs)
    feats["MNF"] = mean_frequency(x, fs)
    feats["LMF"] = log_mean_frequency(x, fs)

    # F2: Energy + AR
    feats["ENG"] = simple_square_integral(x)
    for i, a in enumerate(autoregressive_coeffs(x)):
        feats[f"AR{i + 1}"] = a

    # F3: Statistical
    feats["SKEW"] = skew(x)
    feats["KURT"] = kurtosis(x)
    feats["DASDV"] = difference_absolute_standard_deviation_value(x)
    feats["DAMV"] = difference_absolute_mean_value(x)

    # F4: Histogram moments
    hist, _ = np.histogram(x, bins=20, density=True)
    feats["HIST_MEAN"] = np.mean(hist)
    feats["HIST_STD"] = np.std(hist)
    feats["HIST_SKEW"] = skew(hist)
    feats["HIST_KURT"] = kurtosis(hist)

    # F5: Band powers + LBP
    sbp = spectral_band_powers(x, fs)
    for i, v in enumerate(sbp):
        feats[f"SBP{i + 1}"] = v
    feats["LBP"] = local_binary_pattern_feature(x)

    # F6: RMS + TDPSD
    td_psd_vals = welch(x, fs=fs, nperseg=min(512, len(x)))[1]
    feats["TDPSD"] = np.mean(td_psd_vals)
    feats["DASDV_F6"] = difference_absolute_standard_deviation_value(x)
    feats["DAMV_F6"] = difference_absolute_mean_value(x)

    # F9: invTDD
    invtdd = invariant_time_domain_descriptor(x)
    for i, v in enumerate(invtdd):
        feats[f"invTDD_{i + 1}"] = v

    return feats

# Processing files
def process_file(file_path, save=True):
    df = pd.read_csv(file_path)
    channels = [c for c in df.columns if c.startswith("Chan_")]
    if "Label" not in df.columns:
        raise ValueError(f"No 'Label' column in {file_path}.")
    label_col = "Label"

    n_samples = len(df)
    features_list = []

    for start in range(0, n_samples - WIN_LEN + 1, STEP):
        end = start + WIN_LEN
        segment = df.iloc[start:end]
        label = segment[label_col].mode().iloc[0]

        ch_data = []
        feats_all = {}

        for ch in channels:
            x = segment[ch].values
            x0 = remove_baseline(x, window_size=1000)
            x1 = channel_standardize(x0)

            thr = THR_FACTOR * np.std(x1)
            ch_data.append(x1)

            feats_ch = extract_features(x1, FS, thr)
            for k, v in feats_ch.items():
                feats_all[f"{ch}_{k}"] = v

        # Global / cross-channel features
        feats_all["MeanCorr"] = channel_correlation(ch_data)
        tsd = temporal_spatial_descriptors(ch_data)
        ntsd = normalized_temporal_spatial_descriptors(ch_data)
        for i, v in enumerate(tsd):
            feats_all[f"TSD_{i + 1}"] = v
        for i, v in enumerate(ntsd):
            feats_all[f"NTSD_{i + 1}"] = v

        feats_all["Label"] = label
        features_list.append(feats_all)

    feature_df = pd.DataFrame(features_list)
    if save:
        out_name = os.path.splitext(os.path.basename(file_path))[0] + "_features.csv"
        feature_df.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
    print(f"Processed {file_path}: {len(feature_df)} windows extracted.")
    return feature_df


# Processing all filtered CSV
all_features = []
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith("_filtered.csv"):
        path = os.path.join(DATA_DIR, fname)
        feats = process_file(path)
        feats["Subject"] = os.path.splitext(fname)[0]
        all_features.append(feats)

all_df = pd.concat(all_features, ignore_index=True)
combined_path = os.path.join(OUTPUT_DIR, "all_train_features.csv")
all_df.to_csv(combined_path, index=False)
print(f"\n✅ All subjects processed. Combined features saved to:\n{combined_path}")

# -------------------------------
# plot_emg_features.py
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load features
file_path = "data/processed/train/all_train_features.csv"
df = pd.read_csv(file_path)

# Remove non-feature columns (keep numeric features)
non_features = ["Label"]
features = [col for col in df.columns if col not in non_features]

print(f"Loaded dataset: {df.shape[0]} samples, {len(features)} features")

# Create output folder for plots
os.makedirs("plots/features", exist_ok=True)

# -------------------------------
# 1️⃣ Histogram for each feature
# -------------------------------
for feat in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feat], kde=True, bins=40, color="steelblue")
    plt.title(f"Distribution of {feat}")
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/features/{feat}_histogram.png", dpi=300)
    plt.close()

print("✅ Saved individual histograms in 'plots/features/' folder.")

# -------------------------------
# 2️⃣ Histogram per Label (optional)
# -------------------------------
for feat in features:
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x=feat, hue="Label", kde=True, bins=40, alpha=0.5)
    plt.title(f"{feat} Distribution by Label")
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/features/{feat}_histogram_by_label.png", dpi=300)
    plt.close()

print("✅ Saved histograms per label in 'plots/features/' folder.")

# -------------------------------
# 3️⃣ Boxplot comparison per feature
# -------------------------------
plt.figure(figsize=(14, 7))
melted = df.melt(id_vars="Label", value_vars=features, var_name="Feature", value_name="Value")
sns.boxplot(data=melted, x="Feature", y="Value", hue="Label")
plt.title("Boxplots of All Features by Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/features/all_features_boxplot.png", dpi=300)
plt.close()

print("✅ Saved 'all_features_boxplot.png'.")

# -------------------------------
# 4️⃣ Violin plot (if data not too large)
# -------------------------------
sample_df = df.sample(min(3000, len(df)), random_state=42)  # sample to avoid overload
plt.figure(figsize=(14, 7))
melted = sample_df.melt(id_vars="Label", value_vars=features, var_name="Feature", value_name="Value")
sns.violinplot(data=melted, x="Feature", y="Value", hue="Label", split=True)
plt.title("Violin Plots of Features by Label (sampled)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/features/all_features_violin.png", dpi=300)
plt.close()

print("✅ Saved 'all_features_violin.png'.")

# -------------------------------
# 5️⃣ Pairplot (for feature correlation overview)
# -------------------------------
sample_df = df.sample(min(2000, len(df)), random_state=42)
sns.pairplot(sample_df[features + ["Label"]], hue="Label", diag_kind="kde", corner=True)
plt.savefig("plots/features/pairplot_features.png", dpi=300)
plt.close()

print("✅ Saved 'pairplot_features.png'.")
print("🎨 All plots successfully generated in 'plots/features/'.")

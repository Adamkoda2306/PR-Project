import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load features
file_path = "data/processed/train/all_train_features.csv"
df = pd.read_csv(file_path, nrows=50000)

non_features = ["Label"]
features = ["_MAV", "_RMS", "_VAR", "_WL", "_ZC", "_SSC", "_WAMP", "_SSI"]
all_features = [col for col in df.columns if any(k in col for k in features)]

# Filtering out features with no variance (Constant Features)
features = [
    col for col in all_features 
    if df[col].nunique() > 1
]

print(f"Loaded dataset: {df.shape[0]} samples, {len(features)} features (after filtering constant ones)")

# Create output folder for plots
os.makedirs("plots/features", exist_ok=True)

# Histogram for each feature 
for feat in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feat], kde=True, bins='auto', color="steelblue") 
    plt.title(f"Distribution of {feat}")
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/features/{feat}_histogram.png", dpi=300)
    plt.close()

print("✅ Saved individual histograms in 'plots/features/' folder.")


# Histogram per Label
for feat in features:
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x=feat, hue="Label", kde=True, bins='auto', alpha=0.5) 
    plt.title(f"{feat} Distribution by Label")
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"plots/features/{feat}_histogram_by_label.png", dpi=300)
    plt.close()

print("✅ Saved histograms per label in 'plots/features/' folder.")

# Boxplot comparison per feature
plt.figure(figsize=(14, 7))
melted = df.melt(id_vars="Label", value_vars=features, var_name="Feature", value_name="Value")
sns.boxplot(data=melted, x="Feature", y="Value", hue="Label")
plt.title("Boxplots of All Features by Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/features/all_features_boxplot.png", dpi=300)
plt.close()

print("✅ Saved 'all_features_boxplot.png'.")

# 4️⃣ Violin plot
sample_df = df.sample(min(3000, len(df)), random_state=42)
plt.figure(figsize=(14, 7))
melted = sample_df.melt(id_vars="Label", value_vars=features, var_name="Feature", value_name="Value")
sns.violinplot(data=melted, x="Feature", y="Value", hue="Label", split=True)
plt.title("Violin Plots of Features by Label (sampled)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/features/all_features_violin.png", dpi=300)
plt.close()

print("✅ Saved 'all_features_violin.png'.")


# 5️⃣ Pairplot
sample_df = df.sample(min(2000, len(df)), random_state=42)
# Uses the filtered 'features' list
sns.pairplot(sample_df[features + ["Label"]], hue="Label", diag_kind="kde", corner=True)
plt.savefig("plots/features/pairplot_features.png", dpi=300)
plt.close()

print("✅ Saved 'pairplot_features.png'.")
print("🎨 All plots successfully generated in 'plots/features/'.")
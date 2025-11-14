import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load Data
file_path = "data/processed/train/all_train_features.csv"
df = pd.read_csv(file_path, nrows=50000)

# features and labels
f0_keywords = ["_MAV", "_RMS", "_VAR", "_WL", "_ZC", "_SSC", "_WAMP", "_SSI"]
features = [col for col in df.columns if any(k in col for k in f0_keywords)]
X = df[features].values
y = df["Label"].values
y = y - y.min()

print(f"Loaded {df.shape[0]} samples with {len(features)} features.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# output folder
os.makedirs("plots/RF", exist_ok=True)


# Correlation Heatmap
corr = pd.DataFrame(X, columns=features).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap (Random Forest)")
plt.tight_layout()
plt.savefig("plots/RF/feature_correlation_heatmap.png", dpi=300)
plt.close()
print("✅ Saved 'feature_correlation_heatmap.png'")

# PCA Visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratios:", pca.explained_variance_ratio_)

# 2D plot
plt.figure(figsize=(7, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y,
                palette="tab10", s=25, alpha=0.7)
plt.title("PCA (2 Components) - Random Forest")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)")
plt.tight_layout()
plt.savefig("plots/RF/pca_2d.png", dpi=300)
plt.close()

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=y, cmap="tab10", s=15, alpha=0.8)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("PCA (3 Components) - Random Forest")
plt.legend(*scatter.legend_elements(), title="Label")
plt.tight_layout()
plt.savefig("plots/RF/pca_3d.png", dpi=300)
plt.close()
print("✅ Saved 'pca_2d.png' and 'pca_3d.png'")

# Random Forest Classifier + CV
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring="accuracy", n_jobs=-1)

print("\n📊 5-Fold Cross-Validation Results (Random Forest):")
print(f"Fold Accuracies: {[f'{score*100:.2f}%' for score in cv_scores]}")
print(f"Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std Deviation: {cv_scores.std()*100:.2f}%")

# Confusion Matrix + Report
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = model.score(X_test, y_test)
print(f"\nHold-out Test Accuracy: {acc*100:.2f}%")

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("plots/RF/confusion_matrix.png", dpi=300)
plt.close()
print("✅ Saved 'confusion_matrix.png'")

# Classification Report
report = classification_report(y_test, y_pred)
print("\n📄 Classification Report:\n", report)

with open("plots/RF/classification_report.txt", "w") as f:
    f.write("Random Forest Classifier Report\n\n")
    f.write(report)
print("✅ Saved 'classification_report.txt'")

# Feature Importance (Random Forest)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.barh(np.array(features)[sorted_idx], importances[sorted_idx])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/RF/random_forest_feature_importance.png", dpi=300)
plt.close()

# Save CSV
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("plots/RF/feature_importance_rf.csv", index=False)

print("✅ Saved 'random_forest_feature_importance.png' and 'feature_importance_rf.csv'")

print("\n🎨 All Random Forest analysis complete.")

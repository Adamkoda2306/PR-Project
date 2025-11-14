import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance


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
os.makedirs("plots/XGBoost", exist_ok=True)


# Feature Correlation Heatmap
corr = pd.DataFrame(X, columns=features).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap (XGBoost)")
plt.tight_layout()
plt.savefig("plots/XGBoost/feature_correlation_heatmap.png", dpi=300)
plt.close()
print("✅ Saved 'feature_correlation_heatmap.png'")


# PCA Visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratios:", pca.explained_variance_ratio_)

# 2D PCA plot
plt.figure(figsize=(7, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="tab10", s=25, alpha=0.7)
plt.title("PCA (2 Components) Visualization - XGBoost")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Var)")
plt.tight_layout()
plt.savefig("plots/XGBoost/pca_2d.png", dpi=300)
plt.close()

# 3D PCA plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                     c=y, cmap="tab10", s=15, alpha=0.8)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("PCA (3 Components) Visualization - XGBoost")
plt.legend(*scatter.legend_elements(), title="Label")
plt.tight_layout()
plt.savefig("plots/XGBoost/pca_3d.png", dpi=300)
plt.close()
print("✅ Saved 'pca_2d.png' and 'pca_3d.png'")



# XGBoost Classifier + Cross Validation
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softmax" if len(np.unique(y)) > 2 else "binary:logistic",
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

# ---- 5-Fold Cross Validation ----
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring="accuracy", n_jobs=-1)

print("\n📊 5-Fold Cross-Validation Results (XGBoost):")
print(f"Fold Accuracies: {[f'{score*100:.2f}%' for score in cv_scores]}")
print(f"Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std Deviation: {cv_scores.std()*100:.2f}%")


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)
print(f"\nHold-out Test Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (XGBoost)")
plt.tight_layout()
plt.savefig("plots/XGBoost/confusion_matrix.png", dpi=300)
plt.close()
print("✅ Saved 'confusion_matrix.png'")

# Classification Report
report = classification_report(y_test, y_pred)
print("\n📄 Classification Report:\n", report)

with open("plots/XGBoost/classification_report.txt", "w") as f:
    f.write("XGBoost Classifier Report\n\n")
    f.write(report)
print("✅ Saved 'classification_report.txt'")


# 5️⃣ Feature Importance (Gain-based)
plt.figure(figsize=(10, 8))
plot_importance(model, importance_type="gain", title="Feature Importance (XGBoost - Gain)",
                xlabel="Gain", ylabel="Feature", grid=False, color="green")
plt.tight_layout()
plt.savefig("plots/XGBoost/xgboost_feature_importance.png", dpi=300)
plt.close()

# Save importance as CSV
importance_df = pd.DataFrame({
    "Feature": model.get_booster().get_score(importance_type="gain").keys(),
    "Gain Importance": model.get_booster().get_score(importance_type="gain").values()
})
importance_df = importance_df.sort_values(by="Gain Importance", ascending=False)
importance_df.to_csv("plots/XGBoost/feature_importance_xgboost.csv", index=False)

print("✅ Saved 'xgboost_feature_importance.png' and 'feature_importance_xgboost.csv'")

print("🎨 All XGBoost analysis complete.")
print("Generated files:")
print(" - feature_correlation_heatmap.png")
print(" - pca_2d.png, pca_3d.png")
print(" - confusion_matrix.png")
print(" - classification_report.txt")
print(" - xgboost_feature_importance.png")
print(" - feature_importance_xgboost.csv")

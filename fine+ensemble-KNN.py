import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

print("🚀 Loading dataset...")
file_path = "data/processed/train/all_train_features.csv"
df = pd.read_csv(file_path, nrows=50000)

# -----------------------------
# Select only F0 EMG features
# -----------------------------
f0_keywords = ["_MAV", "_RMS", "_VAR", "_WL", "_ZC", "_SSC", "_WAMP", "_SSI"]
features = [col for col in df.columns if any(k in col for k in f0_keywords)]

X = df[features].values
y = df["Label"].values
y = y - y.min()

# -----------------------------
# Train–Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]
subspace_size = min(60, num_features)

# -----------------------------
# ⭐ Fine + Ensemble KNN (BEST)
# -----------------------------
fine_ensemble_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(
        n_neighbors=13,
        metric="manhattan",
        weights="distance"
    ),
    n_estimators=20,
    max_features=subspace_size,
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=-1
)

fine_ensemble_knn.fit(X_train, y_train)
y_pred = fine_ensemble_knn.predict(X_test)

# -----------------------------
# Performance
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 FINAL ACCURACY (Fine + Ensemble KNN):", accuracy)
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Fine + Ensemble KNN")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("fine_ensemble_knn_confusion_matrix.png", dpi=300)
plt.close()

print("📌 Saved confusion matrix: fine_ensemble_knn_confusion_matrix.png")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(fine_ensemble_knn, "Fine_Ensemble_KNN.pkl")
print("💾 Saved model: Fine_Ensemble_KNN.pkl")
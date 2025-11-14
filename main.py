# ============================================================
# main.py — Full EMG Processing Pipeline
# ============================================================
# Steps:
# 1. Noise Reduction
# 2. Feature Extraction
# 3. Plot Generation
# 4. Train SVM (Cubic Kernel)
# 5. Train Random Forest
# 6. Train XGBoost
# 7. Train Fine + Ensemble KNN
# ============================================================

import os
import subprocess

VENV_PYTHON = r"./env/Scripts/python.exe"

def run(script):
    print("\n" + "=" * 60)
    print(f"▶️ Running: {script}")
    print("=" * 60)

    result = subprocess.run([VENV_PYTHON, script])

    if result.returncode != 0:
        print(f"❌ ERROR while running {script}")
        exit(1)

    print(f"✅ Completed: {script}\n")



# ============================================================
# PIPELINE EXECUTION
# ============================================================

print("\n\n🚀 Starting Full EMG Pipeline...\n")

# ---------------------------------------
# 1. Noise Reduction
# ---------------------------------------
run("noise-remove.py")

# ---------------------------------------
# 2. Feature Extraction
# ---------------------------------------
run("feature-extraction.py")

# ---------------------------------------
# 3. Plots
# ---------------------------------------
# run("plots.py")

# ---------------------------------------
# 4. SVM — Cubic Kernel
# ---------------------------------------
run("svm-cubic.py")

# ---------------------------------------
# 5. Random Forest
# ---------------------------------------
run("randomforest.py")

# ---------------------------------------
# 6. XGBoost
# ---------------------------------------
run("xgboost-train.py")

# ---------------------------------------
# 7. Fine + Ensemble KNN
# ---------------------------------------
run("fine+ensemble-KNN.py")

print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
print("📌 Pipeline: Noise → Features → Plots → SVM → RF → XGB → KNN")

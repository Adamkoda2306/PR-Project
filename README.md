PR Project — EMG Signal Classification Pipeline
============================================================

This project implements a complete end-to-end machine learning pipeline 
for EMG (Electromyography) signal processing using the EMAHA-DB1 dataset.

The system performs:
1. Noise Removal
2. Window-Based Feature Extraction (F0–F9 Features)
3. Data Visualization and Plots
4. Training Machine Learning Models:
   - SVM (Cubic Kernel)
   - Random Forest
   - XGBoost
   - Fine + Ensemble KNN

The entire pipeline can be executed automatically using: main.py


DATASET (REQUIRED)
============================================================

Download the dataset from this link:

EMAHA-DB1 Dataset:
https://www.kaggle.com/datasets/anishturlapaty/emaha-db1

After downloading:
1. Extract the dataset.
2. Rename the extracted folder to: EMG-dataset
3. Place it inside the project root directory:

PR-Project/
    EMG-dataset/
    main.py
    noise-remove.py
    feature-extraction.py
    plots.py
    requirements.txt
    svm-cubic.py
    randomforest.py
    xgboost-train.py
    fine+ensemble-KNN.py



PROJECT PIPELINE
============================================================

1. Noise Reduction
   - 50 Hz Notch Filter
   - 500 Hz Low Pass Filter
   - Time-domain and PSD comparison plots

2. Feature Extraction
   Window size: 250 ms
   Overlap: 50%
   Extracted features:
   - F0: MAV, RMS, VAR, WL, ZC, SSC, WAMP, SSI
   - F1: MNF, MDF, LMF
   - F2–F9: Advanced features including AR, SBP, TSD, invTDD, histogram features.

   Output:
   data/processed/train/all_train_features.csv

3. Visualization
   - Histograms
   - Boxplots
   - Violin plots
   - Pairplots
   - PCA Plots
   - Confusion Matrices

4. Machine Learning Models
   - SVM (Cubic Kernel)
   - Random Forest
   - XGBoost
   - Fine + Ensemble KNN

Generated results are stored inside:
plots/
data/processed/



INSTALLATION GUIDE
============================================================

1. Clone the repository:

git clone https://github.com/Adamkoda2306/PR-Project
cd PR-Project

2. Create a virtual environment:

Windows:
```bash
python -m venv env
./env/Scripts/Activate
```

Mac/Linux:
```bash
python3 -m venv env
source env/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```



RUN THE FULL PIPELINE
============================================================

Inside the virtual environment, run:

python main.py

The pipeline will execute all steps automatically:
1. Noise Removal
2. Feature Extraction
3. Plot Generation
4. SVM Training
5. Random Forest Training
6. XGBoost Training
7. Fine + Ensemble KNN Training


RECOMMENDED PROJECT STRUCTURE
============================================================

PR-Project/
    EMG-dataset/
    data/
        train/
        processed/train/
    plots/
    main.py
    noise-remove.py
    feature-extraction.py
    plots.py
    svm-cubic.py
    randomforest.py
    xgboost-train.py
    fine+ensemble-KNN.py



TROUBLESHOOTING
============================================================

1. Seaborn Module Not Found:
   Activate the virtual environment:
   env\Scripts\activate
   pip install seaborn

2. Git Large File Error (HTTP 408):
   Use Git LFS for large files:
   git lfs install
   git lfs track "*.csv"
   git add .
   git commit -m "Add LFS tracking"
   git push origin main



CREDITS
============================================================

Dataset:
EMAHA-DB1 EMG Dataset
https://www.kaggle.com/datasets/anishturlapaty/emaha-db1

Project Team Members:
Koda Adam
Goluguri Nikhil Suri Reddy
Abraham Scariya
Kalla Sanjay Naidu

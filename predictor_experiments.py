# predictor_experiments.py
# ------------------------------------------------------------------
# Experimental version of predictor.py for testing accuracy improvements
# Adds richer features, tuning, and evaluation — safe to modify
# ------------------------------------------------------------------

import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
import pandas as pd
from pathlib import Path

# -----------------------------------------------------
# Field class
# -----------------------------------------------------
class field:
    def __init__(self, dapiPath: str, transPath: str, groupNum: int, treatNum: int):
        self.dapiImg = cv.imread(dapiPath)
        self.transImg = cv.imread(transPath)
        self.groupNum = groupNum
        self.treatNum = treatNum


# -----------------------------------------------------
# Dataset loader
# -----------------------------------------------------
def loadDataset(dataDir: str):
    dataset = []
    path = Path(dataDir)
    datasetNames = {"_".join(f.stem.split("_")[:-1]) for f in path.glob("*.jpg")}

    for sharedName in datasetNames:
        dapiPath = str(path / f"{sharedName}_DAPI.jpg")
        transPath = str(path / f"{sharedName}_TRANS.jpg")

        if not Path(dapiPath).exists() or not Path(transPath).exists():
            print(f"[WARNING] Missing file for {sharedName}")
            continue

        parts = sharedName.split("_")
        groupNum = int(parts[0][1:])
        treatNum = int(parts[4][1:])
        dataset.append(field(dapiPath, transPath, groupNum, treatNum))
    return dataset


# -----------------------------------------------------
# Feature extractors
# -----------------------------------------------------
def cellCountFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    num_labels, _ = cv.connectedComponents(binary)
    return num_labels - 1


def circularityFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    circ = [
        (4 * np.pi * cv.contourArea(c)) / (cv.arcLength(c, True) ** 2 + 1e-5)
        for c in contours
        if cv.arcLength(c, True) > 0
    ]
    return np.mean(circ) if circ else 0


def relaSizeFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contours]
    return np.mean(areas) if areas else 0


def brightnessFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    return np.mean(gray[gray > 0])


# ---------- New features ----------
def textureContrastFeature(data: field):
    """Approximate texture contrast using Laplacian variance."""
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(gray, cv.CV_64F)
    return np.var(lap)


def intensityVarianceFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    return np.var(gray)


def cellDensityFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) < 3:
        return 0
    pts = np.vstack([c.squeeze() for c in contours if len(c) > 4])
    hull = ConvexHull(pts)
    return len(contours) / hull.area


def objectLevelFeatures(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=gray)
    if not props:
        return [0, 0, 0]
    areas = [p.area for p in props]
    ecc = [p.eccentricity for p in props]
    intensities = [p.mean_intensity for p in props]
    return [np.mean(areas), np.mean(ecc), np.mean(intensities)]

# ------------------------------------------------------
# Segmentation function (12 unit task) 
# ------------------------------------------------------
def segmentCells(data: field):
    # Preprocess image
    imgRGB = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)

    # Histogram-based thresholding
    _, imgThreshold = cv.threshold(gray, 0, 255, 
                                   cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Morphological dilation
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv.morphologyEx(imgThreshold, cv.MORPH_DILATE, kernel)

    # Distance transform
    distTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 0)
    
    # Normalize to 0–255 and convert to 8-bit
    distTrans_norm = cv.normalize(distTrans, None, 0, 
                                  255, cv.NORM_MINMAX).astype('uint8')

    # Second thresholding on distance
    _, distThresh = cv.threshold(distTrans_norm, 0, 255,
                                 cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Connected components
    distThresh = np.uint8(distThresh)
    num_labels, labels = cv.connectedComponents(distThresh)

    # Overlay watershed boundaries (labels == -1)
    segmented = imgRGB.copy()
    segmented[labels == -1] = [255, 0, 0]  # Red boundary lines

    # Update image in place
    data.dapiImg = segmented

    return

# -----------------------------------------------------
# Helper to compute all features
# -----------------------------------------------------
def buildFeatureMatrix(dataset):
    X, y_treat, y_group, y_day = [], [], [], []
    for d in dataset:
        feats = [
            cellCountFeature(d),
            circularityFeature(d),
            relaSizeFeature(d),
            brightnessFeature(d),
            textureContrastFeature(d),
            intensityVarianceFeature(d),
            cellDensityFeature(d),
            *objectLevelFeatures(d),
        ]
        X.append(feats)
        y_treat.append(d.treatNum)
        y_group.append(d.groupNum)
        if d.groupNum <= 5:
            y_day.append(1)
        else:
            y_day.append(2)
    return np.array(X), np.array(y_treat), np.array(y_group), np.array(y_day)


# -----------------------------------------------------
# Model training and evaluation
# -----------------------------------------------------
def trainRF(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTraining RandomForest for {label} ...")

    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        params,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{label} Accuracy: {acc * 100:.2f}%")
    print("Best params:", grid.best_params_)
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{label} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    Path("results").mkdir(exist_ok=True)
    plt.savefig(f"results/{label}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    importances = clf.feature_importances_
    feat_names = [
        "CellCount",
        "Circularity",
        "RelSize",
        "Brightness",
        "Texture",
        "Variance",
        "Density",
        "ObjArea",
        "ObjEcc",
        "ObjIntensity",
    ][: len(importances)]

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    imp_df.sort_values("Importance", ascending=False).to_csv(
        f"results/{label}_feature_importance.csv", index=False
    )

    sns.barplot(
        data=imp_df.sort_values("Importance", ascending=False),
        x="Importance",
        y="Feature",
        color="teal",
    )
    plt.title(f"{label} Feature Importances")
    plt.tight_layout()
    plt.savefig(f"results/{label}_feature_importance_plot.png", bbox_inches="tight")
    plt.close()

    return clf


# -----------------------------------------------------
# Main execution
# -----------------------------------------------------
if __name__ == "__main__":
    data = loadDataset("./train_data")
    X, y_treat, y_group, y_day = buildFeatureMatrix(data)

    print("\n=== Treatment Model ===")
    trainRF(X, y_treat, "Treatment")

    print("\n=== Group Model ===")
    trainRF(X, y_group, "Group")

    print("\n=== Day Model ===")
    trainRF(X, y_day, "Day")

    print("\n=== Segmentation Treatment Model ===")
    for d in data:
        segmentCells(d)
    trainRF(X, y_treat, "Segmentation_Treatment")

    print("\nFinished all experiments. Results saved in ./results/")

import os
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
import pandas as pd
from shutil import copy2


# -----------------------------------------------------
# Field class
# -----------------------------------------------------
class field:
    def __init__(self, dapiPath: str, transPath: str, groupNum: int, treatNum: int):
        self.dapiImg = cv.imread(dapiPath)
        self.transImg = cv.imread(transPath)
        self.dapiPath = dapiPath
        self.transPath = transPath
        self.groupNum = groupNum
        self.treatNum = treatNum


# -----------------------------------------------------
# Dataset loader
# -----------------------------------------------------
def loadDataset(dataDir: str):
    dataset = []
    path = Path(dataDir)
    datasetNames = {"_".join(f.stem.split("_")[:-1]) for f in path.glob("*.jpg")}

    # Grab DAPI and TRANS file path name for each field image
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
# Feature extractor helpers
# -----------------------------------------------------

# Number of cells in a DAPI image
def cellCountFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    num_labels, _ = cv.connectedComponents(binary)
    return num_labels - 1

# Average circularity of all cells in a TRANS image
def circularityFeature(data: field):
    gray = cv.cvtColor(data.transImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # average the circularity of each contour
    circ = [
        (4 * np.pi * cv.contourArea(c)) / (cv.arcLength(c, True) ** 2 + 1e-5)
        for c in contours
        if cv.arcLength(c, True) > 0
    ]
    return np.mean(circ) if circ else 0

# Average area of all nuclei in DAPI image
def relaSizeFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(c) for c in contours]
    return np.mean(areas) if areas else 0

# Average brightness of a DAPI image
def brightnessFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    return np.mean(gray[gray > 0])

# Sharpness of edges of DAPI image
def textureContrastFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    # emphasize local contrast + texture variation
    lap = cv.Laplacian(gray, cv.CV_64F)
    return np.var(lap)

# Variance of brightness of a DAPI image
def intensityVarianceFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    return np.var(gray)

# How tightly packed cells are in a DAPI image
def cellDensityFeature(data: field):
    gray = cv.cvtColor(data.dapiImg, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) < 3:
        return 0
    pts = np.vstack([c.squeeze() for c in contours if len(c) > 4])

    # smallest convex polygon containing all cells
    hull = ConvexHull(pts)

    # cell density = num cells / area of smallest convex polygon containing all of them
    return len(contours) / hull.area

# Features for each cell: average area, eccentricity, and intensity across all cels
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
# Segmentation function (12 unit) 
# ------------------------------------------------------
def segmentCells(img):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Otsu thresholding
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Optional: remove small noise with morphology
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=1)

    # Find contours (segmented cell boundaries)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.drawContours(imgRGB, contours, -1, (255, 0, 0), 1)  # red outlines

    return imgRGB

# -----------------------------------------------------
# Helper to compute all features
# -----------------------------------------------------
def buildFeatureMatrix(dataset):
    X, y_treat, y_group, y_day, names = [], [], [], [], []
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
        names.append(Path(d.dapiPath).name)
    return np.array(X), np.array(y_treat), np.array(y_group), np.array(y_day), np.array(names)


# -----------------------------------------------------
# Model training and evaluation
# -----------------------------------------------------
def trainRF(X, y, label, names):
    # split into train and test groups
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42
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

    # misclassification tracking
    mis_idx = np.where(y_pred != y_test)[0]
    print(f"{len(mis_idx)} misclassified samples out of {len(y_test)}")

    Path("results/misclassified").mkdir(parents=True, exist_ok=True)
    for i in mis_idx:
        src_path = Path("seg_data" if "Segmentation" in label else "train_data") / names_test[i]
        dst_path = Path("results/misclassified") / f"{label}_true{y_test[i]}_pred{y_pred[i]}_{names_test[i]}"
        if src_path.exists():
            copy2(src_path, dst_path)
    if len(mis_idx) > 0:
        print(f"Misclassified examples saved in results/misclassified/")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{label} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    Path("results").mkdir(exist_ok=True)
    plt.savefig(f"results/{label}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # feature importance
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
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    data = loadDataset("./train_data")
    X, y_treat, y_group, y_day, names = buildFeatureMatrix(data)

    # training + testing each model
    print("\n=== Treatment Model ===")
    trainRF(X, y_treat, "Treatment", names)

    print("\n=== Group Model ===")
    trainRF(X, y_group, "Group", names)

    print("\n=== Day Model ===")
    trainRF(X, y_day, "Day", names)

    print("\n=== Segmentation Treatment Model ===")
    
    input_dir = "./train_data"
    output_dir = "./seg_data"
    os.makedirs(output_dir, exist_ok=True)

    # Segment images
    # for filename in os.listdir(input_dir):
    #     if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
    #         continue
    #     img_path = os.path.join(input_dir, filename)
    #     img = cv.imread(img_path)
    #     segmented_img = segmentCells(img)
    #     out_path = os.path.join(output_dir, filename)
    #     cv.imwrite(out_path, cv.cvtColor(segmented_img, cv.COLOR_RGB2BGR))

    dataSeg = loadDataset("./seg_data")
    X_seg, y_treat, _, _, names = buildFeatureMatrix(dataSeg)
    trainRF(X_seg, y_treat, "Segmentation_Treatment", names)

    print("\nFinished all experiments. Results saved in ./results/")

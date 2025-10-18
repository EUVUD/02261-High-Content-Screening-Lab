import cv2 as cv
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Data structure to hold images
class field:
    dapiImg = None
    transImg = None
    groupNum = 0 #Which group data belongs to
    treatNum = 0 #Which treatment applies on the data

    def __init__(self, dapiPath : str, transPath : str, groupNum : int, 
                 treatNum : int):
        self.dapiImg = cv.imread(dapiPath)
        self.transImg = cv.imread(transPath)
        self.groupNum = groupNum
        self.treatNum = treatNum

# Dataset Loader
def loadDataset(dataDir : str):
    
    datasetName = set()
    dataset = []

    path = Path(dataDir)
    for file in path.glob('*.jpg'):
        nameParts = file.stem.split('_')[:-1] #Separate by '_' and remove last part
        sharedParts = '_'.join(nameParts) #Join back to get shared name
        datasetName.add(sharedParts)

    for sharedName in datasetName:
        # Extract paths
        dapiPath = str(path / f"{sharedName}_DAPI.jpg")
        transPath = str(path / f"{sharedName}_TRANS.jpg")

        # Load images
        dapiImg = cv.imread(dapiPath)
        transImg = cv.imread(transPath)

        nameParts = sharedName.split('_')
        groupNum = int(nameParts[0][1:])
        treatNum = int(nameParts[4][1:])

        dataset.append(field(dapiPath, transPath, groupNum, treatNum))

    return dataset


def cellCountFeature(data : field): #Use watershed to count cells
    img = data.dapiImg
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Reduce noise
    img = cv.GaussianBlur(img, (11, 11), 0)

    #Apply edge detection
    img = cv.Canny(img, 30, 150, 3)
    
    #Connect edges
    img = cv.dilate(img, (1, 1), iterations=0)

    (cnt, hierarchy) = cv.findContours(
    img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    return len(cnt)

def circularityFeature(data : field): #Use the circularity formula
    img = data.transImg

    #Convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Apply Otsu's thresholding
    ret, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

    #Find contours
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    circularities = []
    for cont in contours:
        contArea = cv.contourArea(cont)
        contPeri = cv.arcLength(cont, True)
        if contPeri > 0:
            circ = (4 * np.pi * contArea) / (contPeri ** 2)
            circularities.append(circ)
        else:
            circularities.append(0)

    return np.mean(circularities)

def relaSizeFeature(data : field): #Use Area / Number of cells
    img = data.transImg

    num_cells = cellCountFeature(data)

    #Convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #Find contours
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    areas = []
    for cont in contours:
        contArea = cv.contourArea(cont)
        areas.append(contArea)

    total_area = np.sum(areas)

    if num_cells == 0:
        return 0
    else:
        rela_size = total_area / num_cells

    return rela_size

def brightnessFeature(data : field): #Use the mean brightness
    img = data.dapiImg

    #Convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Pick pixels with intensity > 0
    bright_pixels = img[img > 0]

    mean_brightness = np.mean(bright_pixels)
    return mean_brightness


imgDataset = loadDataset('./train_data')

#Treatment prediction model

X = []
y = []

for data in imgDataset:
    features = [
        cellCountFeature(data),
        circularityFeature(data),
        relaSizeFeature(data),
        brightnessFeature(data)
    ]
    X.append(features)
    y.append(data.treatNum)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

trtClassifier = RandomForestClassifier(n_estimators=100, random_state=42)
trtClassifier.fit(X_train, y_train)
y_pred = trtClassifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
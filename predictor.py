import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks

class field:
    dapiImg = None
    transImg = None

    def __init__(self, dapiPath : str, transPath : str):
        self.dapiImg = cv.imread(dapiPath)
        self.transImg = cv.imread(transPath)

def cellCountFeature(data : field): #Use watershed to count cells
    img = data.dapiImg
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    hist = cv.calcHist([img.astype('float32')], [0], None, [256], [0, 256])
    hist = hist.ravel()
    hist_smooth = cv.GaussianBlur(hist.reshape(-1,1), (5,1), 0).ravel()
    
    peaks, properties = find_peaks(hist_smooth, height=np.max(hist_smooth)*0.1, distance=20)
    peak_heights = hist_smooth[peaks]
    highest_peak_idx = peaks[np.argmax(peak_heights)]

    thresh_val = highest_peak_idx + 10

    _, imgThreshold = cv.threshold(img,thresh_val,255,cv.THRESH_BINARY)

    kernal = np.ones((3, 3), np.uint8)
    imgDilate = cv.morphologyEx(imgThreshold, cv.MORPH_DILATE, kernal)

    distTrans = cv.distanceTransform(imgDilate, cv.DIST_L2, 0)

    hist = cv.calcHist([distTrans.astype('float32')], [0], None, [256], [0, 256])
    hist = hist.ravel()
    hist_smooth = cv.GaussianBlur(hist.reshape(-1,1), (5,1), 0).ravel()
    first_peak_idx = np.argmax(hist_smooth[:50])
    thresh_val = first_peak_idx + 5

    _, distThresh = cv.threshold(distTrans, thresh_val, 255, cv.THRESH_BINARY)

    distThresh = np.uint8(distThresh)
    num_labels, labels = cv.connectedComponents(distThresh)

    labels = np.int32(labels)
    labels = cv.watershed(imgRGB, labels)

    imgRGB[labels == -1] = [255, 0, 0]

    return num_labels

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
        circ = (4 * np.pi * contArea) / (contPeri ** 2)
        circularities.append(circ)

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
#Takes in the board image as input, does preprocessing and chops it into tiles

from __future__ import print_function
import cv2
import numpy as np
import torch
import torch.nn as nn 

#Displays the grid on top of the warped image...
def displayGrid(image):
    cell_height = image.shape[0] // 9
    cell_width = image.shape[1] // 9
    indentation = 0
    rects = []
    
    for i in range(9):
        for j in range(9):
            p1 = (j*cell_height + indentation, i*cell_width + indentation)
            p2 = ((j+1)*cell_height - indentation, (i+1)*cell_width - indentation)
            rects.append((p1, p2))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
    return rects

#Preprocesses the input image...
def preprocessImage(image, skip_dilation = False):
    preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocess = cv2.GaussianBlur(preprocess, (9, 9), 0)
    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if not skip_dilation:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)
        preprocess = cv2.dilate(preprocess, kernel, iterations = 1)

    return preprocess

#Gets the coords of the corner points of largest rectangle...
def getCoords(image):
    contours, _ = cv2.findContours(preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = all_contours[0]
    sums = []
    diffs = []
    
    for point in polygon:
        for x, y in point:
            sums.append(x + y)
            diffs.append(x - y)
            
    top_left = polygon[np.argmin(sums)].squeeze()
    bottom_right = polygon[np.argmax(sums)].squeeze() 
    top_right = polygon[np.argmax(diffs)].squeeze()
    bottom_left = polygon[np.argmin(diffs)].squeeze() 
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)

#Warping the image...
def warp(image, coords):
    ratio = 1.2
    tl, tr, br, bl = coords
    widthA = np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[1])**2)
    widthB = np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[1])**2)
    heightA = np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[1])**2)
    heightB = np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[1])**2)
    width = max(widthA, widthB) * ratio
    height = width
    
    destination = np.array([
        [0, 0],
        [height, 0],
        [height, width],
        [0, width]], dtype = np.float32)
    M = cv2.getPerspectiveTransform(coords, destination)
    warped = cv2.warpPerspective(image, M, (int(height), int(width)))
    return warped

#Extracting each grid image...
def extractGrid(image, rects):
    tiles = []
    for coords in rects:
        rect = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        tiles.append(rect)
    return tiles

filepath = raw_input("Enter an image filepath : ")
image = cv2.imread(filepath)
preprocess = cv2.resize(preprocessImage(image), (600, 600))
preprocess = cv2.bitwise_not(preprocess, preprocess)

coords = getCoords(preprocess)
preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
coordsImage = preprocess.copy()

for coord in coords:
    cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)

warpedImage = warp(preprocess, coords)
rects = displayGrid(warpedImage)
tiles = extractGrid(warpedImage, rects)

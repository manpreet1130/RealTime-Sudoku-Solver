import cv2
import numpy as np


def displayGrid(image):
    cell_height = image.shape[0] // 9
    cell_width = image.shape[1] // 9
    indentation = 0
    rects = []
    '''
    for i in range(9):
        cv2.rectangle(image, (0, i * width + indentation), (image.shape[0], i*width + indentation), (0, 255, 0), thickness = 3)
        cv2.rectangle(image, (i * height + indentation, 0), (i * height + indentation, image.shape[1]), (0, 255, 0), thickness = 3)
    '''
    for i in range(9):
        for j in range(9):
            p1 = (j*cell_height + indentation, i*cell_width + indentation)
            p2 = ((j+1)*cell_height - indentation, (i+1)*cell_width - indentation)
            rects.append((p1, p2))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
    return rects

def preprocessImage(image, skip_dilation = False):
    preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocess = cv2.GaussianBlur(preprocess, (9, 9), 0)
    #_, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #adaptiveMean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    if not skip_dilation:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)
        preprocess = cv2.dilate(preprocess, kernel, iterations = 1)

    return preprocess

def getCoords(image):
    contours, _ = cv2.findContours(preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = all_contours[0]
    #print(polygon[0].shape)
    #print(type(polygon[0]))
    #print(polygon[0])
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
    #print(polygon[top_left], polygon[top_right], polygon[bottom_left], polygon[bottom_right])
    #return polygon[1:5]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype = np.float32)

def warp(image, coords):
    ratio = 1
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

def extractGrid(image, rects):
    tiles = []
    for coords in rects:
        rect = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        tiles.append(rect)
    return tiles


image = cv2.imread('data/sudoku5.jpeg')
preprocess = cv2.resize(preprocessImage(image), (600, 600))
preprocess = cv2.bitwise_not(preprocess, preprocess)

#external_contours, _ = cv2.findContours(preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, _ = cv2.findContours(preprocess, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print(getCoords(preprocess))
coords = getCoords(preprocess)
#print(coords.shape)
#top_left, top_right, bottom_left, bottom_right = getCoords(preprocess)
preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
coordsImage = preprocess.copy()

for coord in coords:
    cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)

#print(preprocess.shape)
warpedImage = warp(preprocess, coords) 
rects = displayGrid(warpedImage)
tiles = extractGrid(warpedImage, rects)
#tile = tiles[0]
#print(len(tiles))
#preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
for tile in tiles:
    pass
#all_contours = cv2.drawContours(preprocess.copy(), contours, -1, (255, 0, 0), thickness = 2)
#external_only = cv2.drawContours(preprocess.copy(), external_contours, -1, (255, 0, 0), thickness = 2)

while True:
    cv2.imshow("Normal", image)
    #cv2.imshow("Gray", gray)
    #cv2.imshow("Binary", binary)
    #cv2.imshow("Adaptive Gaussian", adaptiveGaussian)
    #cv2.imshow("Adaptive Mean", adaptiveMean)
    #cv2.imshow("Blur", blur)
    #cv2.imshow("Preprocessed", preprocess)
    cv2.imshow("Coords", coordsImage)
    cv2.imshow("Warped", warpedImage)
    #cv2.imshow("Tile", tiles[0])
    #cv2.imshow("tile", tile)
    #cv2.imshow("All Contours", all_contours)
    #cv2.imshow("External Contours", external_only)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
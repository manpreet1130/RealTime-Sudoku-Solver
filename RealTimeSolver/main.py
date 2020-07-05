import cv2
from basic import *
from sud import *
from solver import *

cap = cv2.VideoCapture(0)

if __name__ == "__main__":
	detected = False
	solved = False
	tiles = []
	print("Get board closer to webcam until stated otherwise...")
	while True:
		retr, frame = cap.read()
		preprocess = preprocessImage(frame)
		preprocess = cv2.bitwise_not(preprocess.copy(), preprocess.copy())
		contourImage = preprocess.copy()
		contourImage = cv2.cvtColor(contourImage, cv2.COLOR_GRAY2BGR)
		coordsImage = contourImage.copy()
		contours, polygon = getContours(preprocess)
		#print(cv2.contourArea(polygon))
		#print(cv2.contourArea(polygon))
		coords = getCoords(contourImage, polygon)
		#print(coords)
		#print(cv2.contourArea(polygon))
		if detected and solved:
			unwarpedImage = unwarp(solutionImage, coords)
		else:
			unwarpedImage = np.zeros((frame.shape[0], frame.shape[1]))
			
		if cv2.contourArea(polygon) > 80000 and not detected:
			#coords = getCoords(contourImage, polygon)
			for coord in coords:
				cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)
				#cv2.circle(frame, (coord[0], coord[1]), 5, (255, 0, 0), -1)
			cv2.drawContours(contourImage, polygon, -1, (0, 255, 0), 3)
			cv2.drawContours(frame, polygon, -1, (0, 255, 0), 3)
			warpedImage = warp(coordsImage.copy(), coords)
			warpedImage = cv2.resize(warpedImage, (540, 540))
			rects = displayGrid(warpedImage)
			tiles = extractGrid(warpedImage, rects)
			if cv2.contourArea(polygon) >= 90000:
				print("Detected")
				detected = True
				#cv2.imwrite('./frame.png', frame)
				#cv2.imwrite('./preprocess.png', preprocess)
				#cv2.imwrite('./contour.png', contourImage)
				#cv2.imwrite('./coords.png', coordsImage)
				#solved = False
			else:
				print("Bring closer...")
			#for i, tile in enumerate(tiles):
			#	cv2.imwrite('./tiles/' + str(i) + '.png', tile)
		
		else:
			#print("Show puzzle...")
			warpedImage = np.zeros((540, 540))
			
		if detected and not solved:
				predictions = getPredictions(tiles)
				#print(predictions)
				solutionImage = solveSudoku(predictions, coords)
				#cv2.imwrite('./solution.png', solutionImage)
				solved = True
				
		
		if retr == True:
			cv2.imshow("Frame", frame)
			#cv2.imshow("Final", final)
			#cv2.imshow("Preprocess", preprocess)
			#cv2.imshow("PreprocessNot", preprocessNot)
			#cv2.imshow("Contour Image", contourImage)
			#cv2.imshow("Coords Image", coordsImage)
			#cv2.imshow("Warped Image", warpedImage)
			#cv2.imshow("Unwarped", unwarpedImage)
			if solved:
				cv2.imshow("Solution", solutionImage)
				
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break
			
		else:
			cap.release()
			cv2.destroyAllWindows()
			break

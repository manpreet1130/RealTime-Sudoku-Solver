import cv2
from basic import *
from sud import *
from solver import *


if __name__ == "__main__":
	images = []
	predictions = []
	filepath = raw_input("Enter an image filepath : ")
	image = cv2.imread(filepath)
	preprocess = cv2.resize(preprocessImage(image), (540, 540))
	preprocess = cv2.bitwise_not(preprocess, preprocess)

	coords = getCoords(preprocess)
	preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
	coordsImage = preprocess.copy()

	for coord in coords:
	    cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)

	warpedImage = warp(preprocess, coords)
	rects = displayGrid(warpedImage)
	tiles = extractGrid(warpedImage, rects)

	for i, tile in enumerate(tiles):
		preprocess = preprocessImage(tile)
		flag, centered = centeringImage(preprocess)
		centeredImage = cv2.resize(centered, (32, 32))
		images.append(centeredImage)
		centeredImage = torch.Tensor(centeredImage).unsqueeze(dim = 0).unsqueeze(dim = 0)
	
		preds = model(centeredImage)
		_, prediction = torch.max(preds, dim = 1)
		if flag:
			predictions.append(prediction.item() + 1)
		else:
			predictions.append(0)
			
	board = np.array(predictions).reshape((9, 9))
	print(board)
	print("Solving...")
	solver = SudokuSolver(board)
	solver.solve()
	final = solver.board
	if 0 in final:
		print("Error occured while solving, try another image!")
	else:
		print(final)
		solutionBoard = cv2.imread('./boards/blank.png')
		solutionImage = displaySolution(solutionBoard, final, predictions)
		print("Press 'q' to quit...")
		while True:
			cv2.imshow("Original Image", image)
			#cv2.imshow("Warped Image", warpedImage)
			#cv2.imshow("Coordinates", coordsImage)
			cv2.imshow("Solution", solutionImage)
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

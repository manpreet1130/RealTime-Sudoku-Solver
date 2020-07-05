#Solving the sudoku using the backtracking algorithm and showing the solution!
from __future__ import print_function
import numpy as np
import cv2


class SudokuSolver:
	def __init__(self, board):
		self.board = board
	
	def printBoard(self):
		for i in range(9):
			for j in range(9):
				print(self.board[i][j], end = " ")
			print("\n")
		
	
	def empty(self):
		for i in range(9):
			for j in range(9):
				if self.board[i][j] == 0:
					return i, j
		return None
			
	def isValid(self, num, row, col):
		#Checking row and col
		for i in range(9):
			if self.board[row][i] == num and i != col:
				return False
			if self.board[i][col] == num and i != row:
				return False
		#Checking the boxes
		y = row // 3
		x = col // 3
		for i in range(y*3, y*3 + 3):
			for j in range(x*3, x*3 + 3):
				if self.board[i][j] == num and (i != row or j != col):
					return False
		return True

	def solve(self):
		pos = self.empty()
		if pos == None:
			return True
		else:
			row, col = pos
			
		for i in range(1, 10):
			self.board[row][col] = i
			if self.isValid(i, row, col) and self.solve():
				return True
				
			else:
				self.board[row][col] = 0
			
		return False
			
    
def displaySolution(image, final, predictions):
	image = image.copy()
	cell_width = image.shape[1] // 9
	cell_height = image.shape[0] // 9
	counter = 0
	for i in range(9):
		for j in range(9):
			if predictions[counter] != 0:
				color = (0, 0, 0)
			else:
				color = (255, 0, 0)
			if final[i][j] == 0:
				print("Couldn't properly solve!")
				return None
			
			text = str(final[i][j])
			offsetX = cell_width // 15
			offsetY = cell_height // 15
			font = cv2.FONT_HERSHEY_SIMPLEX
			(text_height, text_width), baseline = cv2.getTextSize(text, font, fontScale = 1, thickness = 3)
			marginX = cell_width // 7
			marginY = cell_height // 7
			bottomLeft = cell_width*j + (cell_width - text_width) // 2 + offsetX
			bottomRight = cell_height*(i+1) - (cell_height - text_height) // 2 + offsetY
			image = cv2.putText(image, text, (int(bottomLeft), int(bottomRight)), font, 1, color, thickness = 3, lineType = cv2.LINE_AA)
			counter += 1
	return image

					
if __name__ == "__main__":
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
			cv2.imshow("Actual Image", image)
			cv2.imshow("Warped Image", warpedImage)
			cv2.imshow("Coords Image", coordsImage)
			cv2.imshow("Solution", solutionImage)
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

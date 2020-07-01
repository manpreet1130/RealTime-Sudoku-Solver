from __future__ import print_function
import numpy as np
import sud
import cv2
from PIL import Image, ImageDraw, ImageFont


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
			
def displayGrid(board, predictions, inclusive):
	height = 540
	width = 540
	image = np.ones((height, width))
	cell_height = image.shape[0] // 9
	cell_width = image.shape[1] // 9
	for i in range(10):
	    p1 = (i*cell_width, 0)
	    p2 = (i*cell_width, height)
	    if i % 3 == 0:
	    	cv2.line(image, p1, p2, (0, 0, 0), 6)
	    else: cv2.line(image, p1, p2, (0, 0, 0), 1)
	for i in range(10):
		p1 = (0, i*cell_height)
		p2 = (width, i*cell_height)
		if i % 3 == 0:
	    		cv2.line(image, p1, p2, (0, 0, 0), 6)
	    	else: cv2.line(image, p1, p2, (0, 0, 0), 1)	    	
	return image
	
	
					
if __name__ == "__main__":
	predictions = sud.predictions
	inclusive = sud.inclusive
	board = np.array(predictions).reshape((9, 9))
	print(board)
	print("Solving...")
	solver = SudokuSolver(board)
	solver.solve()
	final = solver.board
	print(final)
	print("Press 'q' to quit")
	while True:
		cv2.imshow("Actual Image", sud.basic.image)
		cv2.imshow("Warped Image", sud.basic.warpedImage)
		cv2.imshow("Coords Image", sud.basic.coordsImage)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

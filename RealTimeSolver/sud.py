import numpy as np 
import torch
import torchvision
import cv2
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import christopher as chris


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		self.dropout = nn.Dropout(p = 0.40)
		self.fc1 = nn.Sequential(
			nn.Linear(8*8*64, 128),
			nn.ReLU())
		self.fc2 = nn.Linear(128, 10)

	def forward(self, input):
		out = self.layer1(input)
		out = self.layer2(out)
		out = self.dropout(out)
		out = out.reshape(out.shape[0], -1)
		out = self.fc1(out)
		out = self.dropout(out)
		out = F.softmax(self.fc2(out), dim = 1)
		return out


model = Network()
try:
	print("Importing model...")
	model.load_state_dict(torch.load('./christopher.pt'))
except:
	print("Training model...")
	chris.train()

model.eval()

def preprocessImage(image, skip_dilation = False):
	preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)

	if not skip_dilation:
		kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype = np.uint8)
		preprocess = cv2.dilate(preprocess, kernel)
	return preprocess

def centeringImage(image):
	rows = image.shape[0]
	for i in range(rows):
		#Floodfilling the outermost layer
		cv2.floodFill(image, None, (0, i), 0)
		cv2.floodFill(image, None, (i, 0), 0)
		cv2.floodFill(image, None, (rows-1, i), 0)
		cv2.floodFill(image, None, (i, rows-1), 0)
		#Floodfilling the penultimate layer
		cv2.floodFill(image, None, (1, i), 0)
		cv2.floodFill(image, None, (i, 1), 0)
		cv2.floodFill(image, None, (rows-2, i), 0)
		cv2.floodFill(image, None, (i, rows-2), 0)


	top = None
	bottom = None
	left = None
	right = None
	threshold = 50
	center = rows // 2
	
	for i in range(center, rows):
		if bottom is None:
			temp = image[i]
			if sum(temp) < threshold or i == rows - 1:
				bottom = i
		if top is None:
			temp = image[rows - i - 1]
			if sum(temp) < threshold or i == rows - 1:
				top = rows - i - 1
		if left is None:
			temp = image[:, rows - i - 1]
			if sum(temp) < threshold or i == rows - 1:
				left = rows - i - 1
		if right is None:
			temp = image[:, i]
			if sum(temp) < threshold or i == rows - 1:
				right = i
	if (top == left and bottom == right):
		return 0, image

	image = image[top - 5:bottom + 5, left - 5:right + 5]
	return 1, image    


def getPredictions(tiles):
	images = []
	predictions = []

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
	return predictions



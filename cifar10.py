import numpy as np 
import torch
import torchvision
import cv2
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

#Network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.dropout = nn.Dropout(p = 0.40)
        self.fc1 = nn.Sequential(
            nn.Linear(7*7*64, 128),
            nn.ReLU())
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = F.softmax(self.fc2(out), dim = 1)
        return out

model = Network()
model.load_state_dict(torch.load('./data/conv_net.pt'))
model.eval()


def preprocessImage(image, skip_dilate = False):
    preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #preprocess = cv2.GaussianBlur(preprocess, (3, 3), 0)
    #preprocess = cv2.normalize(preprocess, preprocess, 0, 255, cv2.NORM_MINMAX)
    #preprocess = cv2.medianBlur(preprocess, 5)
    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)
    #preprocess = preprocess - np.mean(preprocess)
    #preprocess = cv2.threshold(preprocess, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if not skip_dilate:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)
        preprocess = cv2.dilate(preprocess, kernel, iterations = 1)

    return preprocess
    
def centeringImage(image):
    rows = image.shape[0]
    for i in range(rows):
        #Floodfilling the outermost layer
        cv2.floodFill(image, None, (0, i), 0)
        cv2.floodFill(image, None, (i, 0), 0)
        cv2.floodFill(image, None, (rows-1, i), 0)
        cv2.floodFill(image, None, (i, rows-1), 0)
        #Floodfilling the second outermost layer
        '''
        cv2.floodFill(image, None, (1, i), 1)
        cv2.floodFill(image, None, (i, 1), 1)
        cv2.floodFill(image, None, (rows - 2, i), 1)
        cv2.floodFill(image, None, (i, rows - 2), 1)
        '''
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
    #cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), thickness = 2)
    #cv2.circle(image, (right, bottom), 3, (255, 255, 255), -1)
    #cv2.circle(image, (left, top), 3, (255, 255, 255), -1)
    #cv2.circle(image, (right, top), 3, (255, 255, 255), -1)
    #cv2.circle(image, (left, bottom), 3, (255, 255, 255), -1)
    image = image[top - 5:bottom + 5, left - 5:right + 5]
    return 1, image    
#images = []
preds = []
for i in range(81):
    image = cv2.imread('./data/images/' + str(i) + '.jpg')
    preprocess = preprocessImage(image)
    flag, centered = centeringImage(preprocess.copy())
    if flag:
        centeredCopy = torch.Tensor(cv2.resize(centered, (28, 28))).unsqueeze(dim = 0).unsqueeze(dim = 0)
        #images.append(centered)

        pred = model(centeredCopy)
        #print(pred)
        _, prediction = torch.max(pred, dim = 1)
        preds.append(prediction.item())
    else:
        preds.append(0)
print(preds)
'''
image = cv2.imread('./data/images/33.jpg')
preprocess = preprocessImage(image)
centered = centeringImage(preprocess.copy())
centeredCopy = torch.Tensor(cv2.resize(centered, (28, 28))).unsqueeze(dim = 0).unsqueeze(dim = 0)
pred = model(centeredCopy)
_, prediction = torch.max(pred, dim = 1)
print(prediction.item())
'''
while True:
    cv2.imshow("Preprocess", preprocess)
    cv2.imshow("Centered", centered)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

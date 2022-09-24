"""
This file will:
- load the saved model
- take in a test png image and predict its class
"""
import torch
from model import SimpleNet
import cv2
import matplotlib.pyplot as plt
from glob import glob
from torchvision import datasets, transforms
import numpy as np


model = SimpleNet()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

def pre_process(image):
	# gray to binary
	# resize
	image = cv2.resize(image, dsize=(28, 28))
	# convert to tensor
	image = torch.tensor(image)
	# reshape to C, H, W; add channel dimension
	image = image.unsqueeze(dim=0).unsqueeze(dim=0).float()
	# apply transforms
	# image = transform(image)
	# image = image / image.mean()

	return image

def find_white_background(imgArr, threshold=0.3):
    """find images with transparent or white background"""
    background = np.array([255])
    percent = (imgArr == background).sum() / imgArr.size
    if percent >= threshold:
        print(percent)
        return True
    else:
        return False


path_list = glob('test/*.png')
num = len(path_list)
size = int(num**0.5)+1
fig = plt.figure(figsize=(size, size))
ax = []
for i, file in enumerate(path_list):
	image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	(thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	is_inverted = find_white_background(image)
	print('backg', is_inverted)
	if is_inverted: image = cv2.bitwise_not(image)
	# preprocess image
	image_proc = pre_process(image)
	output = model(image_proc)
	pred_class = output.argmax(dim=1, keepdim=True)
	# do inference
	# import pdb; pdb.set_trace()

	ax.append(fig.add_subplot(size, size, i+1))
	ax[-1].set_title(f'Pred: {pred_class.item()}')
	plt.imshow(image)
fig.tight_layout()
plt.show()
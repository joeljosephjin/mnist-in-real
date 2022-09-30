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
import onnxruntime as rt



class DoInference():
	def __init__(self, use_onnx=False):
		self.use_onnx = use_onnx
		if self.use_onnx:
			self.sess = rt.InferenceSession("mnist_cnn.onnx")
			self.input_name = self.sess.get_inputs()[0].name
			self.label_name = self.sess.get_outputs()[0].name
		else:
			self.model = SimpleNet()
			self.model.load_state_dict(torch.load('mnist_cnn.pt'))
			self.model.eval()

		self.transform = transforms.Compose([
		    # transforms.ToTensor(),
		    transforms.Normalize((0.1307,), (0.3081,))
		    ])

	def model_onnx(self, image):
		pred = self.sess.run([self.label_name], {self.input_name: np.array(image).astype(np.float32)})[0]
		return pred

	def pre_process(self, image):
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

	def find_white_background(self, imgArr, threshold=0.3):
	    """find images with transparent or white background"""
	    background = np.array([255])
	    percent = (imgArr == background).sum() / imgArr.size
	    if percent >= threshold:
	        print(percent)
	        return True
	    else:
	        return False

	def get_image_prediction(self, image):
		(thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		# reverse color of image if its of white background and not black
		is_inverted = self.find_white_background(image)
		print('backg', is_inverted)
		if is_inverted: image = cv2.bitwise_not(image)
		# preprocess image
		image_proc = self.pre_process(image)
		if self.use_onnx:
			output = self.model_onnx(image_proc)
		else:
			output = self.model(image_proc)
		pred_class = output.argmax(dim=1, keepdim=True)

		return pred_class

	def run_on_folder(self, folder='test'):
		path_list = glob(f'{folder}/*.png')
		num = len(path_list)
		size = int(num**0.5)+1
		fig = plt.figure(figsize=(size, size))
		ax = []
		for i, file in enumerate(path_list):
			image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

			pred_class = self.get_image_prediction(image)

			# do inference
			ax.append(fig.add_subplot(size, size, i+1))
			ax[-1].set_title(f'Pred: {pred_class.item()}')
			plt.imshow(image)
		fig.tight_layout()
		plt.show()


if __name__=="__main__":
	inference = DoInference(use_onnx=False)
	# inference = DoInference(use_onnx=True)
	inference.run_on_folder('test')
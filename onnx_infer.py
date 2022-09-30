import onnxruntime as rt
import numpy as np
from model import SimpleNet
import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np


class DoInference():
	def __init__(self):

		self.sess = rt.InferenceSession("mnist_cnn.onnx")
		self.input_name = self.sess.get_inputs()[0].name
		self.label_name = self.sess.get_outputs()[0].name

	def model_onnx(self, image):
		pred = self.sess.run([self.label_name], {self.input_name: np.array(image).astype(np.float32)})[0]
		return pred

	def pre_process(self, image):
		# gray to binary
		# resize
		image = cv2.resize(image, dsize=(28, 28))
		image = image.reshape(1, 1, 28, 28)

		return image

	def find_white_background(self, imgArr, threshold=0.3):
	    """find images with transparent or white background"""
	    background = np.array([255])
	    percent = (imgArr == background).sum() / imgArr.size
	    if percent >= threshold:
	        # print(percent)
	        return True
	    else:
	        return False

	def get_image_prediction(self, image):
		(thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		# reverse color of image if its of white background and not black
		is_inverted = self.find_white_background(image)
		# print('backg', is_inverted)
		if is_inverted: image = cv2.bitwise_not(image)
		# preprocess image
		image_proc = self.pre_process(image)
		output = self.model_onnx(image_proc)
		# output = self.model(image_proc)

		# import pdb; pdb.set_trace()
		pred_class = output.argmax()

		return pred_class

	def run_on_folder(self, folder='test', show=True):
		path_list = glob(f'{folder}/*.png')
		num = len(path_list)
		size = int(num**0.5)+1
		if show:
			fig = plt.figure(figsize=(size, size))
			ax = []
		for i, file in enumerate(path_list):
			image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

			pred_class = self.get_image_prediction(image)

			print('pred:', pred_class, '; True:', file.split('/')[-1][:-4])

			if show:
				ax.append(fig.add_subplot(size, size, i+1))
				ax[-1].set_title(f'Pred: {pred_class.item()}')
				plt.imshow(image)
		if show:
			fig.tight_layout()
			plt.show()


if __name__=="__main__":
	inference = DoInference()
	inference.run_on_folder('test', show=False)
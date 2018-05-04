import Classification
import FileReader
from FileReader import get_detection_images
from CaseManager import collect_and_enhance
from sklearn.externals import joblib
import ImageEnhancer
import timeit
import numpy as np
import cv2
from skimage import io, filters, exposure

letters = 'abcdefghijklmnopqrstuvwxyz'

class Detector: # main class for handeling detection using sliding window and ANN for prediction
	def __init__(self):  
		self.images=get_detection_images() #load from FileReader

	def set_picture(self,num): #(200,200) and (300,600) two pictures
		self.current_image=self.images[num]

	def train_ANN(self): #training neural networks with examples from chars74k-lite
		self.classification=Classification.Classifier()
		self.classification.train_neural_network()
		_ = joblib.dump(self.classification.network, "classification_weights.pkl", compress=9)
		# storing to file for mobility in testing

	def load_weights(self): #used for mobility in testing
		self.classification=Classification.Classifier()
		self.classification.network = joblib.load("classification_weights.pkl")

	def get_frames(self): # creates frames using sliding window detection
		graytones = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
		ret, tresh = cv2.threshold(graytones, 252, 255, cv2.THRESH_BINARY_INV)
		image, ctrs, hier = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		self.rects = [cv2.boundingRect(ctr) for ctr in ctrs]
		for rect in self.rects:
			cv2.rectangle(self.current_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

	def generate_patches(self): # generates patches for testing
		detected_images = []
		for rect in self.rects:
			x_1, y_1, x_size, y_size = rect
			x_2 = x_1+x_size
			y_2 = y_1+y_size
			temp_image = []
			rows = self.current_image[y_1:y_2]
			for row in rows:
				temp_image.append(row[x_1:x_2])
			detected_images.append(temp_image)
		self.detected_images = np.array(detected_images)

	def predict_images(self): #predicts patch classification based on neural network
		preds = []
		for i, im in enumerate(self.detected_images):
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = ImageEnhancer.enhance_image_neural_net(im)
			input_vector = []
			for row in im:
				input_vector.extend(row)
			pred = self.classification.neural_network_prediction([input_vector])
			pred = letters[pred[0]]
			preds.append((self.rects[i], pred))

		for rect, pred in preds:
			cv2.putText(self.current_image, pred, (rect[0], rect[1]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
		io.imshow(self.current_image)
		io.show()
	
	def run(self): # main method for running detection and prediction
		self.train_ANN()
		self.set_picture(0)
		self.get_frames()
		self.generate_patches()
		self.predict_images()
		self.set_picture(1)
		self.get_frames()
		self.generate_patches()
		self.predict_images()

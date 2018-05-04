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

class Detector:
	def __init__(self,size_x=20,size_y=20,step=4):
		self.images=get_detection_images()
		self.size_x=size_x
		self.size_y=size_y
		self.subsamples=[] #list of as cases

	def set_picture(self,num): #(200,200) and (300,600)
		self.current_image=self.images[num]
		#self.frame_x,self.frame_y=self.current_image.shape

	def train_ANN(self):
		self.classification=Classification.Classifier()
		self.classification.train_neural_network()
		print("Saving weights")
		_ = joblib.dump(self.classification.network, "classification_weights.pkl", compress=9)
		print("Done saving weights")

	def load_weights(self): #used for mobility in testing
		self.classification=Classification.Classifier()
		self.classification.network = joblib.load("classification_weights.pkl")

	def test_ANN(self):
		self.test_results=self.classification.neural_network_prediction(self.subsamples)

	def get_frames(self): # potentially change variable names
		graytones = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
		ret, im_th = cv2.threshold(graytones, 252, 255, cv2.THRESH_BINARY_INV)
		image, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		self.rects = [cv2.boundingRect(ctr) for ctr in ctrs]
		self.rects = sorted(self.rects, key = lambda x: (x[0] + 20*x[1]))
		for rect in self.rects:
			cv2.rectangle(self.current_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

	def generate_patches(self): 
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

	def predict_images(self):
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
	
	def run(self):
		self.train_ANN()
		self.set_picture(0)
		self.get_frames()
		self.generate_patches()
		self.predict_images()
		self.set_picture(1)
		self.get_frames()
		self.generate_patches()
		self.predict_images()

def test():
	d=Detector()
	d.load_weights()
	d.set_picture(1)
	d.get_frames()
	d.generate_patches()
	d.predict_images()
	return d
#should train on full set

#TODO: Implement sliding windows algorithm
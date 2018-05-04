import Classification
import FileReader
from FileReader import get_detection_images
from CaseManager import collect_and_enhance
import ImageEnhancer
import timeit
import numpy as np

class Detector:
	def __init__(self,size_x=20,size_y=20,step=4):
		self.images=get_detection_images()
		self.size_x=size_x
		self.size_y=size_y
		self.subsamples=[] #list of as cases

	def set_picture(self,num): #(200,200) and (300,600)
		self.current_image=self.images[num]
		self.frame_x,self.frame_y=self.current_image.shape

	def train_ANN(self):
		self.classification=Classification.Classifier()
		self.classification.train_neural_network()

	def test_ANN(self):
		self.test_results=self.classification.neural_network_prediction(self.subsamples)

	def generate_sub_samples(self):
		subsamples=np.array()
		self.indtocor=[]
		for x_corner in range(self.frame_x-self.size_x):
			for y_corner in range(self.frame_y-self.size_y):
				np.hstack(subsamples,self.current_image[x_corner:x_corner+self.size_x,y_corner:y_corner+self.size_y]) #can possibly be paralellized more
				self.indtocor.append([x_corner,y_corner])
		self.subsamples=collect_and_enhance(subsamples)

		#convert subsamples to matrices so that whole sample may be used for input


	def detect_letters(self): #picture is a 1 or 0
		pass


def test():
	d=Detector()
	print(timeit.default_timer())
	d.set_picture(0)
	print(timeit.default_timer())
	d.generate_sub_samples()
	print(timeit.default_timer())
	d.train_ANN()
	print(timeit.default_timer())
	d.test_ANN()
	print(timeit.default_timer())
	return d
#should train on full set

#TODO: Implement sliding windows algorithm
import Classification
import FileReader


class Detector:
	def __init__(self,image,size_x=20,size_y=20):
		self.main_image=image
		self.size_x=size_x
		self.size_y=size_y

	def generate_sub_samples(self):
		pass

def main():
	#classifier=Classification.Classifier()
	test_1,test_2=get_detection_images()
	print(test_2.shape)

#should train on full set

#TODO: Implement sliding windows algorithm
import Classification
import Detection
import FileReader


def main():
	classification=Classification.Classifier() #Classification part
	classification.do_knn()
	classification.do_svm()
	classification.do_random_forest()
	classification.do_neural_network()
	detection=Detection.Detector() #sliding window and classification
	detection.run()

main()
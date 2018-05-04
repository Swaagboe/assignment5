import Classification
import Detection
import FileReader


def main():
	classification=Classification.Classifier()
	classification.do_knn()
	classification.do_svm()
	classification.do_random_forest()
	classification.do_neural_network()
	detection=Detection.Detector()
	detection.run()

main()
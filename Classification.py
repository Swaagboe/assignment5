import CaseManager as cm


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics


class Classifier:

    def __init__(self):
        case_manager = cm.CaseManager()
        self.train_input, self.train_target = case_manager.get_training_input_matrix_and_targets()
        self.test_input, self.test_target = case_manager.get_test_input_matrix_and_targets()
        a = 0

    def do_knn(self):
        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(self.train_input, self.train_target)
        prediction = knn.predict(self.test_input)
        conf_matrix = confusion_matrix(self.test_target, prediction)
        accuracy = sklearn.metrics.accuracy_score(self.test_target, prediction)
        print("KNN accuracy: " + str(accuracy))
        a=0

    def do_svm(self):
        support_vector_machine = SVC(gamma=0.001)
        support_vector_machine.fit(self.train_input, self.train_target)
        prediction = support_vector_machine.predict(self.test_input)
        conf_matrix = confusion_matrix(self.test_target, prediction)
        accuracy = sklearn.metrics.accuracy_score(self.test_target, prediction)
        print("SVM accuracy: " + str(accuracy))
        a=0

    def do_random_forest(self):
        random_forest = RandomForestClassifier(n_estimators=52)
        random_forest.fit(self.train_input, self.train_target)
        prediction = random_forest.predict(self.test_input)
        conf_matrix = confusion_matrix(self.test_target, prediction)
        accuracy = sklearn.metrics.accuracy_score(self.test_target, prediction)
        print("Random forest accuracy: " + str(accuracy))
        a=0

    def do_neural_network(self):
        neural_network =







c = Classifier()
c.do_knn()
#c.do_svm()
c.do_random_forest()
a = 0










class Algorithm1:

    def __init__(self):
        pass


# Second algorithm for letter classification
class Algorithm2:
    def __init__(self):
        pass

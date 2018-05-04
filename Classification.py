import CaseManager as cm


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.neural_network import MLPClassifier


class Classifier:

    def __init__(self):
        case_manager = cm.CaseManager()
        case_manager.initialize()
        self.train_input, self.train_target = case_manager.get_training_input_matrix_and_targets()
        self.test_input, self.test_target = case_manager.get_test_input_matrix_and_targets()

        case_manager_neural_net = cm.CaseManager()
        case_manager_neural_net.initialize_for_neural_net()
        self.train_input_neural_net, self.train_target_neural_net = case_manager_neural_net.get_training_input_matrix_and_targets()
        self.test_input_neural_net, self.test_target_neural_net = case_manager_neural_net.get_test_input_matrix_and_targets()

        a = 0

    def do_knn(self):
        knn = KNeighborsClassifier(n_neighbors=15)
        knn.fit(self.train_input, self.train_target)
        prediction = knn.predict(self.test_input)
        conf_matrix = confusion_matrix(self.test_target, prediction)
        accuracy = sklearn.metrics.accuracy_score(self.test_target, prediction)
        print("KNN accuracy: " + str(accuracy))
        a=0

    def do_svm(self):
        support_vector_machine = SVC(gamma = 0.003)
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
        # potentially make network variable into intrinsic
        network = MLPClassifier(hidden_layer_sizes=(350,160,50)) #do not change, optimalized
        network.fit(self.train_input_neural_net, self.train_target_neural_net)
        prediction = network.predict(self.test_input)
        accuracy = sklearn.metrics.accuracy_score(self.test_target_neural_net, prediction)
        print("Accuracy neural net: " + str(accuracy))

    def train_neural_network(self):
        self.network = MLPClassifier(hidden_layer_sizes=(350,160,50)) #do not change, optimalized
        self.network.fit(self.train_input_neural_net, self.train_target_neural_net)

    def neural_network_prediction(self, input):
        return self.network.predict(input)












class Algorithm1:

    def __init__(self):
        pass


# Second algorithm for letter classification
class Algorithm2:
    def __init__(self):
        pass

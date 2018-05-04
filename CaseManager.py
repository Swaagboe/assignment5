import FileReader
import Case
import ImageEnhancer
from random import shuffle



#A casemanager that has control over test, training and validation-set
class CaseManager:

    def __init__(self):
        self.cases = [] # list of case-class-obj.
        self.training_cases = []
        self.test_cases= []


    def initialize(self):
        self.collect_cases()
        self.split_into_training_and_test_set()

    def initialize_for_neural_net(self):
        self.collect_cases_for_neural_net()
        self.split_into_training_and_test_set()

    def collect_cases(self):
        images = FileReader.get_all_photos()
        for letter_type in range(len(images)):
            self.cases.append([])
            for image in range(len(images[letter_type])):
                target = letter_type
                photo = images[letter_type][image]
                enhanced_photo = ImageEnhancer.enhance_image(photo)
                case = Case.Case(enhanced_photo, target)
                self.cases[letter_type].append(case)

    def collect_cases_for_neural_net(self): #Only used for neural nw
    # reads file, enhances images and distributes to list
        images = FileReader.get_all_photos()
        for letter_type in range(len(images)):
            self.cases.append([])
            for image in range(len(images[letter_type])):
                target = letter_type
                photo = images[letter_type][image]
                enhanced_photo = ImageEnhancer.enhance_image_neural_net(photo)
                case = Case.Case(enhanced_photo, target)
                self.cases[letter_type].append(case)

    def split_into_training_and_test_set(self): # 20 % of each letter
        for letter_cases in self.cases:
            shuffle(letter_cases)
            number_of_test_cases = int(round(len(letter_cases)*0.2))
            for i in range(number_of_test_cases):
                test_case = letter_cases[i]
                self.test_cases.append(test_case)
            for i in range(number_of_test_cases, len(letter_cases) -1 ):
                training_case = letter_cases[i]
                self.training_cases.append(training_case)

    def get_training_input_matrix_and_targets(self): #adopt to framework
        input_matrix = []
        targets_vector = []
        for case in self.training_cases:
            input_matrix.append(case.input_vector)
            targets_vector.append(case.target)

        return input_matrix, targets_vector

    def get_test_input_matrix_and_targets(self): #adopt to framework
        input_matrix = []
        targets_vector = []
        for case in self.test_cases:
            input_matrix.append(case.input_vector)
            targets_vector.append(case.target)

        return input_matrix, targets_vector

def collect_and_enhance(photoList):# static method, takes the list of test data and creates object to put into NN
    input_matrix = []
    for i in range(len(photoList)):
        enhanced_photo = ImageEnhancer.enhance_image_neural_net(photoList[i])
        input_matrix.append(enhanced_photo)
    return input_matrix




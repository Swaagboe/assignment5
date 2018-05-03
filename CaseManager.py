import FileReader
import Case
import ImageEnhancer
from random import shuffle



#A casemanager that has control over test, training and validationset
class CaseManager:

    def __init__(self):
        self.cases = []
        self.training_cases = []
        self.test_cases= []
        self.initialize()


    def initialize(self):
        self.collect_cases()
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

    def split_into_training_and_test_set(self):
        for letter_cases in self.cases:
            shuffle(letter_cases)
            number_of_test_cases = int(round(len(letter_cases)*0.2))
            for i in range(number_of_test_cases):
                test_case = letter_cases[i]
                self.test_cases.append(test_case)
            for i in range(number_of_test_cases, len(letter_cases) -1 ):
                training_case = letter_cases[i]
                self.training_cases.append(training_case)

    def get_training_input_matrix_and_targets(self):
        input_matrix = []
        targets_vector = []
        for case in self.training_cases:
            input_matrix.append(case.input_vector)
            targets_vector.append(case.target)

        return input_matrix, targets_vector

    def get_test_input_matrix_and_targets(self):
        input_matrix = []
        targets_vector = []
        for case in self.test_cases:
            input_matrix.append(case.input_vector)
            targets_vector.append(case.target)

        return input_matrix, targets_vector






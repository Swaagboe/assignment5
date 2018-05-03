

class Case:

    def __init__(self, photo, target):
        self.input_vector = []
        for row in photo:
            self.input_vector.extend(row)
        self.target = target

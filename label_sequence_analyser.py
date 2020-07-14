class LabelSequenceAnalyser:

    def __init__(self):
        self.last_labels = []

    def put_label(self, label):
        self.last_labels.append(label)
        if len(self.last_labels) > 2:
            self.last_labels = self.last_labels[1:]

    def get_label(self):
        if len(self.last_labels) == 0:
            return None
        elif self.last_labels.count(self.last_labels[0]) == 2:
            return self.last_labels[0]
        else:
            return None

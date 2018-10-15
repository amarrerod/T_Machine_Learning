

from sets import Set
import numpy as np
import io
from tokenizer import Tokenizer
from email_object import Email_Object
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB

class SpamTrainer:
    def __init__(self, training_files):
        self.categories = []
        for category, _ in training_files:
            self.categories.push(category)
        self.to_train = training_files
        self.model = GaussianNB()

    def train(self):
        for category, file in self.to_train:
            email = Email_Object(io.open(file, 'rb'))
            self.categories.push(category)
            for token in Tokenizer.unique_tokenizer(email.body()):
                self.training[category][token] += 1
                self.totals['_all'] += 1
                self.totals[category] += 1
        self.to_train = {}
        self.categories = np.unique(categories)

if __name__ == "__main__":
    print("NADA")
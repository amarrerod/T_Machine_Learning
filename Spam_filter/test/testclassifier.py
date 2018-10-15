
import unittest
import io
from sets import Set
from spam_filter.email_object import Email_Object
from spam_filter.classifier import SpamTrainer

class TestSpamTrainer(unittest.TestCase):
    def setUp(self):
        self.training = [['spam', './test/fixtures/plain.eml'],
                         ['ham', './test/fixtures/small.eml'],
                         ['scram', './test/fixtures/plain.eml']]
        self.trainer = SpamTrainer(self.training)
        file = io.open('./test/fixtures/plain.eml', 'r')
        self.email = Email_Object(file)
    
    def test_multiple_categories(self):
        categories = self.trainer.categories
        expected = [i for i, j in self.training]
        self.assertEqual(categories, expected)
    
    def test_counts_all_at_zero(self):
        for cat in ['_all', 'spam', 'ham', 'scram']:
            self.assertEqual(self.trainer.total_for(cat), 0)

    def test_probability_being_1_over_n(self):
        trainer = self.trainer
        scores = trainer.score(self.email).values()
        self.assertAlmostEqual(scores[0], scores[-1])
        for i in range(len(scores) - 1):
            self.assertAlmostEqual(scores[i], scores[i + 1])

    def test_add_up_to_one(self):
        trainer = self.trainer
        scores = trainer.normalized_score(self.email).values()
        self.assertAlmostEqual(sum(scores), 1)
        self.assertAlmostEqual(scores[0], 1 / 2.0)
        
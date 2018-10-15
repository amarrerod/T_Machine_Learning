import unittest
import io
import re
from context import spam_filter
from spam_filter.email_object import Email_Object

class TestPlainTextEmailObject(unittest.TestCase):
    CLRF = "\n\n"
    def setUp(self):
        self.plain_file = './test/fixtures/plain.eml'
        self.plain_text = io.open(self.plain_file, 'r')
        self.text = self.plain_text.read()
        self.plain_text.seek(0)
        self.plain_email = Email_Object(self.plain_text)
    
    def test_parse_plain_body(self):
        body = self.CLRF.join(self.text.split(self.CLRF)[1:])
        self.assertEqual(self.plain_email.body(), body)
    
    def test_parse_the_subject(self):
        subject = re.search("Subject: (.*)", self.text).group(1)
        self.assertEqual(self.plain_email.subject(), subject)
    
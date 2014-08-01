import logging, os
import numpy as np
import numpy.random as npr
import VocabUtils as vu

logging.basicConfig(format='%(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('./training_text')

result = vu.build_vocab(sentences, min_count=5, down_sample=0.0)


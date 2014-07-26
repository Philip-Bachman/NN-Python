import gensim, logging, os

logging.basicConfig(format='%(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('./training_text')

model = gensim.models.Word2Vec(sentences, size=200, window=6, min_count=10, workers=8)


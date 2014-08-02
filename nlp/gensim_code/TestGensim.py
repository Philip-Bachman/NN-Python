import logging, os
import numpy as np
import numpy.random as npr
import W2VSimple as w2vs

logging.basicConfig(format='%(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

def some_nearest_words(keys_to_words, sample_count, W):
    norms = np.sqrt(np.sum(W**2.0,axis=1,keepdims=1))
    W = W / (norms + 1e-5)
    source_keys = np.zeros((sample_count,)).astype(np.int32)
    neighbor_keys = np.zeros((sample_count, 10)).astype(np.int32)
    all_keys = np.asarray(keys_to_words.keys()).astype(np.int32)
    for s in range(sample_count):
        i = npr.randint(0,all_keys.size)
        source_k = all_keys[i]
        neg_cos_sims = -1.0 * np.sum(W * W[source_k], axis=1)
        sorted_k = np.argsort(neg_cos_sims)
        source_keys[s] = source_k
        neighbor_keys[s,:] = sorted_k[1:11]
    source_words = []
    neighbor_words = []
    for s in range(sample_count):
        source_words.append(keys_to_words[source_keys[s]])
        neighbor_words.append([keys_to_words[k] for k in neighbor_keys[s]])
    return [source_keys, neighbor_keys, source_words, neighbor_words]

sentences = MySentences('./training_text')

model = w2vs.W2VSimple(sentences, alpha=0.002, size=152, window=6, \
                               min_count=1, workers=4, hs=1)
k2w = {}
w2k = {}
for w in model.vocab:
    k = model.vocab[w].index
    k2w[k] = w
    w2k[w] = k

for i in range(1001):
    print("ROUND {0:d}".format(i))
    sentences = MySentences('./training_text')
    model.train(sentences, chunksize=200)
    if ((i > 1) and ((i % 50) == 0)):
        print("============================================================")
        [s_keys, n_keys, s_words, n_words] = some_nearest_words(k2w, 10, model.syn0)
        for w in range(10):
            print("{0:s}: {1:s}".format(s_words[w],", ".join(n_words[w])))


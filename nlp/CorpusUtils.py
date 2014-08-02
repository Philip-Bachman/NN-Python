#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# 
# Code in this file is based on code with made publically available under the
# license given above. Modifications were made by Philip Bachman (in 2014).
#

import os
import sys
import heapq
import time
import itertools
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import numpy as np
import numpy.random as npr
import numba

from six import iteritems, itervalues
from six.moves import xrange

class SentenceFileIterator(object):
    """
    Iterator over all files in some directory.

    The directory passed to this object's constructor should contain only text
    files. The text files will be parsed extremely naively, by simply treating
    '\n' characters as delimiters between sentences/phrases/paragraphs, or
    whatever, and then splitting each chunk of text on white space (i.e. by
    applying *.split().
    """
    def __init__(self, dirname):
        self.dirname = dirname
        return

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class Vocab(object):
    """
    A single vocabulary item, used internally when running build_vocab().

    This object will end up holding pointers to the LUT key, HSM code params,
    and various other useful things for each word in a built vocabulary.
    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)
        return

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"

def build_vocab(sentences, min_count=5, compute_hs_tree=True, \
                compute_ns_table=True, down_sample=0.0):
    """
    Build vocabulary from a sequence of sentences (can be a once-only generator stream).
    Each sentence must be an iterable sequence of hashable objects.

    """
    # scan the given corpus of sentences and count the occurrences of each word
    sentence_no = -1
    raw_vocab = {}
    total_words = 0
    for sentence_no, sentence in enumerate(sentences):
        if sentence_no % 10000 == 0:
            print("PROGRESS: at sentence #%i, processed %i words and %i word types" % \
                (sentence_no, total_words, len(raw_vocab)))
        for word in sentence:
            total_words += 1
            if word in raw_vocab:
                raw_vocab[word].count += 1
            else:
                raw_vocab[word] = Vocab(count=1)
    print("collected %i word types from a corpus of %i words and %i sentences" % \
        (len(raw_vocab), total_words, sentence_no + 1))

    # assign a unique index to each sufficiently frequent word
    #
    # NOTE: If *UNK* is already present in the source files, we will carry it
    # over into the training vocabulary whether or not it meets the frequency
    # threshold. All other unique tokens/words that don't meet the frequency
    # threshold will be treated as if "converted" to *UNK*. The total frequency
    # for *UNK* will thus be the frequency of the "raw" token *UNK* in the
    # source text plus the summed frequencies of all words in the source text
    # that do not meet the frequency threshold on their own. If *UNK* was not
    # present in the source text as a raw token, it will be added to the vocab
    # and will collect the frequencies of all dropped words.
    words_to_vocabs, words_to_keys, keys_to_words = {}, {}, {}
    idx = 0
    unk_count = 0
    for word, v in iteritems(raw_vocab):
        if ((v.count >= min_count) or (word == '*UNK*')):
            # this word meets the frequency threshold or is *UNK*
            v.index = idx
            words_to_vocabs[word] = v
            words_to_keys[word] = idx
            keys_to_words[idx] = word
            idx += 1
        else:
            # this word is too infrequent and is not *UNK*
            unk_count += v.count
    if '*UNK*' in raw_vocab:
        # *UNK* will have been processed in the above loop
        words_to_vocabs['*UNK*'].count += unk_count
    else:
        # *UNK* was not processed by the above loop, so add it now
        words_to_vocabs['*UNK*'] = Vocab(count=unk_count, index=idx)
        words_to_keys['*UNK*'] = idx
        keys_to_words[idx] = '*UNK*'
    print("total %i word types after removing those with count<%s" % \
        (len(words_to_vocabs), min_count))

    # precalculate downsampling thresholds, which are written into the vocab
    # objects in words_to_vocabs
    _precalc_downsampling(words_to_vocabs, down_sample=down_sample)

    result = {}
    result['words_to_vocabs'] = words_to_vocabs
    result['words_to_keys'] = words_to_keys
    result['keys_to_words'] = keys_to_words
    result['hs_tree'] = None
    result['ns_table'] = None
    if compute_hs_tree:
        result['hs_tree'] = create_binary_tree(words_to_vocabs)
    if compute_ns_table:
        # build the table for drawing random words (for negative sampling)
        result['ns_table'] = _make_table(words_to_vocabs, keys_to_words, \
                words_to_keys)
    return result

def _precalc_downsampling(w2v, down_sample=0.0):
    """
    Precalculate each vocabulary item's retention probability.

    Called from `build_vocab()`.
    """
    assert(down_sample >= 0.0)
    sample = (down_sample > 1e-8)
    total_words = sum([v.count for v in itervalues(w2v)])
    for v in itervalues(w2v):
        prob = 1.0
        if sample:
            prob = np.sqrt(down_sample / (v.count / total_words))
        v.sample_prob = min(prob, 1.0)
    return

def _make_table(w2v, k2w, w2k, table_size=100000000, power=0.75):
    """
    Create a table using stored vocabulary word counts for drawing random words
    in the negative sampling training routines.

    Called from `build_vocab()`.

    """
    # table (= list of words) of noise distribution for negative sampling
    vocab_size = len(k2w)
    table = np.zeros((table_size,), dtype=np.uint32)

    # compute sum of all power (Z in paper)
    power_sum = float(sum([w2v[word].count**power for word in w2v]))
    # go through the whole table and fill it up with the word indexes proportional to a word's count**power
    widx = 0
    # normalize count^0.75 by Z
    d1 = w2v[k2w[widx]].count**power / power_sum
    for tidx in xrange(table_size):
        table[tidx] = widx
        if ((float(tidx) / table_size) > d1):
            widx += 1
            d1 += w2v[k2w[widx]].count**power / power_sum
        if widx >= vocab_size:
            widx = vocab_size - 1
    return table


def create_binary_tree(w2v):
    """
    Create a binary Huffman tree using stored vocabulary word counts. Frequent words
    will have shorter binary codes. Called internally from `build_vocab()`.

    The codes (presumably for use in a Hierarchical Softmax Layer) are stored
    directly into the Vocab ojects that are the values in the passed w2v dict.
    """
    # build the huffman tree
    heap = list(itervalues(w2v))
    heapq.heapify(heap)
    for i in xrange(len(w2v) - 1):
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Vocab(count=(min1.count + min2.count), \
                index=(i + len(w2v)), left=min1, right=min2))
    # recurse over the tree, assigning a binary code to each vocabulary word
    if heap:
        max_depth = 0
        stack = [(heap[0], [], [])]
        while stack:
            node, codes, points = stack.pop()
            if node.index < len(w2v):
                # leaf node => store its path from the root
                node.code, node.point = codes, points
                max_depth = max(len(codes), max_depth)
            else:
                # inner node => continue recursion
                points = np.array(list(points) + [node.index - len(w2v)], dtype=np.uint32)
                stack.append((node.left, np.array(list(codes) + [-1.0], dtype=np.float32), points))
                stack.append((node.right, np.array(list(codes) + [1.0], dtype=np.float32), points))
    return

###################################
# TRAINING EXAMPLE SAMPLING UTILS #
###################################

@numba.jit("void(u4[:], i8, i8, i8, u4[:], u4[:], u4[:], u4[:])")
def fast_pair_sample(phrase, max_window, i, repeats, anc_keys, pos_keys, rand_pool, ri):
    phrase_len = phrase.size
    max_key_idx = anc_keys.shape[0]
    for r in range(repeats):
        j = i + r
        a_idx = rand_pool[ri[0]] % phrase_len
        ri[0] += 1
        red_win = (rand_pool[ri[0]] % max_window) + 1
        ri[0] += 1
        c_min = a_idx - red_win
        if (c_min < 0):
            c_min = 0
        c_max = a_idx + red_win
        if (c_max >= phrase_len):
            c_max = phrase_len - 1
        c_span = c_max - c_min + 1
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = c_min + (rand_pool[ri[0]] % c_span)
            ri[0] += 1
        anc_keys[j] = phrase[a_idx]
        pos_keys[j] = phrase[c_idx]
    return

class PosSampler:
    """
    This samples examples positive example pairs each comprising an anchor
    word and a context word in its "skip-gram window".
    """
    def __init__(self, phrase_list, max_window):
        # phrase_list contains the phrases to sample from 
        self.max_window = max_window
        self.phrase_list = phrase_list
        self.phrase_table = self._make_table(self.phrase_list)
        self.pt_size = self.phrase_table.size
        return

    def _make_table(self, p_list, table_size=10000000):
        """
        Create a table for quickly drawing phrase indices in proportion to
        the length of each phrase.

        """
        phrase_count = len(p_list)
        phrase_lens = np.asarray([p.size for p in p_list]).astype(np.float64)
        len_sum = np.sum(phrase_lens)

        table = np.zeros((table_size,), dtype=np.uint32)
        widx = 0
        d1 = phrase_lens[0] / len_sum
        for tidx in xrange(table_size):
            table[tidx] = widx
            if ((float(tidx) / table_size) > d1):
                widx += 1
                d1 += phrase_lens[widx] / len_sum
            if widx >= phrase_count:
                widx = phrase_count - 1
        return table

    def sample(self, sample_count):
        """Draw a sample."""
        anc_keys = np.zeros((sample_count,), dtype=np.uint32)
        pos_keys = np.zeros((sample_count,), dtype=np.uint32)
        phrase_keys = np.zeros((sample_count,), dtype=np.uint32)
        # we will use a "precomputed" table of random ints, to save overhead
        # on calls through numpy.random. the location of the next fresh random
        # int in rand_pool is given by ri[0]
        rand_pool = npr.randint(0, high=self.pt_size, \
                size=(10*sample_count,)).astype(np.uint32)
        ri = np.asarray([0]).astype(np.uint32) # index into rand_pool
        repeats = 5
        while not ((sample_count % repeats) == 0):
            repeats -= 1
        for i in range(0, sample_count, repeats):
            pt_idx = rand_pool[ri[0]]
            ri[0] = ri[0] + 1
            phrase_keys[i] = self.phrase_table[pt_idx]
            fast_pair_sample(self.phrase_list[phrase_keys[i]], self.max_window, \
                             i, repeats, anc_keys, pos_keys, rand_pool, ri)
        # Sample negative examples from self.neg_table
        anc_keys = anc_keys.astype(np.uint32)
        pos_keys = pos_keys.astype(np.uint32)
        phrase_keys = phrase_keys.astype(np.uint32)
        return [anc_keys, pos_keys, phrase_keys]

class NegSampler:
    """
    This samples "contrastive words" for training via negative sampling.
    """
    def __init__(self, neg_table=None, neg_count=10):
        # phrase_list contains the phrases to sample from 
        self.neg_table = neg_table
        self.neg_table.reshape((self.neg_table.size,))
        self.neg_table_size = self.neg_table.size
        self.neg_count = neg_count
        return

    def sample(self, sample_count, neg_count=0):
        if (neg_count == 0):
            neg_count = self.neg_count
        neg_keys = np.zeros((sample_count, neg_count), dtype=np.uint32)
        neg_idx = npr.randint(0, high=self.neg_table_size, size=neg_keys.shape)
        for i in range(neg_keys.shape[1]):
            neg_keys[:,i] = self.neg_table[neg_idx[:,i]]
        neg_keys = neg_keys.astype(np.uint32)
        return neg_keys


if __name__=="__main__":
    sentences = SentenceFileIterator('./training_text')
    result = build_vocab(sentences, min_count=3, down_sample=0.0)




##############
# EYE BUFFER #
##############

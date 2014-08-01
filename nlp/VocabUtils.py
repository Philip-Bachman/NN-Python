#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Modified by Philip Bachman (in 2014)

import logging
import sys
import heapq
import time
import itertools
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import numpy as np

from six import iteritems, itervalues
from six.moves import xrange


class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
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
    words_to_vocabs, words_to_keys, keys_to_words = {}, {}, {}
    idx = 0
    unk_count = 0
    for word, v in iteritems(raw_vocab):
        if v.count >= min_count:
            v.index = idx
            words_to_vocabs[word] = v
            words_to_keys[word] = idx
            keys_to_words[idx] = word
            idx += 1
        else:
            unk_count += v.count
    words_to_vocabs['*UNK*'] = Vocab(count=unk_count)
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
    #if compute_hs_tree:
    #    result['hs_tree'] = create_binary_tree()
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
    table = np.zeros((table_size,), dtype=np.int32)

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
    return


def create_binary_tree(w2v):
    """
    Create a binary Huffman tree using stored vocabulary word counts. Frequent words
    will have shorter binary codes. Called internally from `build_vocab()`.

    """
    # build the huffman tree
    heap = list(itervalues(w2v))
    heapq.heapify(heap)
    for i in xrange(len(w2v) - 1):
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(w2v), left=min1, right=min2))

    # recurse over the tree, assigning a binary code to each vocabulary word
    if heap:
        max_depth, stack = 0, [(heap[0], [], [])]
        while stack:
            node, codes, points = stack.pop()
            if node.index < len(w2v):
                # leaf node => store its path from the root
                node.code, node.point = codes, points
                max_depth = max(len(codes), max_depth)
            else:
                # inner node => continue recursion
                points = array(list(points) + [node.index - len(w2v)], dtype=np.int32)
                stack.append((node.left, array(list(codes) + [0], dtype=np.int32), points))
                stack.append((node.right, array(list(codes) + [1], dtype=np.int32), points))

        logger.info("built huffman tree with maximum node depth %i" % max_depth)




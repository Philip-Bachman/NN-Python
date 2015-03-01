import numpy as np
import numpy.random as npr


def make_key_dicts(word_list, min_freq=2, unk_word='*UNK*'):
    """Make word-to-key and key-to words dicts from word_list."""
    if type(word_list[0]) == type([]):
        word_hist = {}
        for phrase in word_list:
            for w in phrase:
                if not w in word_hist:
                    word_hist[w] = 1
                else:
                    word_hist[w] += 1
    else:
        word_hist = {}
        for w in word_list:
            if not w in word_hist:
                word_hist[w] = 1
            else:
                word_hist[w] += 1
    kept_words = [w for w in word_hist if (word_hist[w] >= min_freq)]
    w2k = {}
    k2w = {}
    for (i, w) in enumerate(kept_words):
        w2k[w] = i
        k2w[i] = w
    unk_key = max(w2k.values()) + 1
    w2k[unk_word] = unk_key
    k2w[unk_key] = unk_word
    return [w2k, k2w]

class STBNode:
    """
    Node structure for use by STBParser.
    """
    # Set of "punctuation-type" characters, for output formatting
    _right_punctuation = set([",","'","''",".","!","?",":",";"])
    _left_punctuation = set(["`","``"])
    def __init__(self, token='', auto_spawn=False):
        self.gold_class = int(token[1])
        self.word = None
        self.lut_key = -1
        self.has_children = False
        self.left_child = []
        self.right_child = []
        # Deal with token
        assert ((token[0] == '(') and (token[-1] == ')'))
        if (token[3] == '('):
            # This node joins a pair of children
            self.has_children = True
            if auto_spawn:
                child_tokens = self.extract_child_tokens(token)
                self.left_child = STBNode(child_tokens[0], True)
                self.right_child = STBNode(child_tokens[1], True)
        else:
            self.word = token[3:-1]
            self.word = self.word.lower()
        return

    def extract_child_tokens(self, token):
        """Extract the pair of child tokens from this parent token."""
        # First extract left token
        par_count = 1
        start_idx = 3
        end_idx = 4
        while (par_count > 0):
            if (token[end_idx] == '('):
                par_count = par_count + 1
            if (token[end_idx] == ')'):
                par_count = par_count - 1
            end_idx = end_idx + 1
        left_token = token[start_idx:end_idx]
        # Then extract right token
        par_count = 1
        start_idx = end_idx + 1
        end_idx = start_idx + 1
        while (par_count > 0):
            if (token[end_idx] == '('):
                par_count = par_count + 1
            if (token[end_idx] == ')'):
                par_count = par_count - 1
            end_idx = end_idx + 1
        right_token = token[start_idx:end_idx]
        child_tokens = [left_token, right_token]
        return child_tokens

    def phrase_str(self):
        """Get a string representing the phrase rooted at this node."""
        phrase = ""
        if self.has_children:
            left_phrase = self.left_child.phrase_str()
            right_phrase = self.right_child.phrase_str()
            if ((right_phrase[0] in STBNode._right_punctuation) or \
                    (right_phrase[0:3] == "n't") or \
                    (left_phrase[-1] in STBNode._left_punctuation)):
                phrase = "{0:s}{1:s}".format(left_phrase, right_phrase)
            else:
                phrase = "{0:s} {1:s}".format(left_phrase, right_phrase)
        else:
            assert (not (self.word is None))
            phrase = "{0:s}".format(self.word)
        return phrase

    def get_all_tokens(self):
        """Get a list of all tokens in the phrase rooted at this node."""
        tokens = []
        if self.has_children:
            tokens.extend(self.left_child.get_all_tokens())
            tokens.extend(self.right_child.get_all_tokens())
        else:
            assert (not (self.word is None))
            tokens = [self.word]
        return tokens

    def set_lut_keys(self, lut_keys, unk_word='*UNK*'):
        """Set numeric look-up-table indices for words in the subtree rooted
        at this node, based on the dict "lut_keys"."""
        if self.has_children:
            self.left_child.set_lut_keys(lut_keys)
            self.right_child.set_lut_keys(lut_keys)
        else:
            assert (not (self.word is None))
            if (self.word in lut_keys):
                # The word here is known by the dict
                self.lut_key = lut_keys[self.word]
            else:
                # The word here is not known by the dict
                self.lut_key = lut_keys[unk_word]
        return

    def get_lutis_and_labels(self):
        """Get phrase labels and lut index sequences for all sub-phrases
        rooted at this node."""
        lut_idx_seqs = []
        labels = []
        if self.has_children:
            left_vals = self.left_child.get_lutis_and_labels()
            right_vals = self.right_child.get_lutis_and_labels()
            # Make the joint lut index sequence for this node in phrase tree
            my_lut_idx_seq = [luti for luti in left_vals[0][-1]]
            my_lut_idx_seq.extend([luti for luti in right_vals[0][-1]])
            # Get the label for this node in phrase tree
            my_label = self.gold_class
            # Augment the lut_idx_seqs and labels lists
            lut_idx_seqs.extend(left_vals[0])
            lut_idx_seqs.extend(right_vals[0])
            lut_idx_seqs.append(my_lut_idx_seq)
            labels.extend(left_vals[1])
            labels.extend(right_vals[1])
            labels.append(my_label)
        else:
            assert (self.lut_key >= 0)
            lut_idx_seqs.append([self.lut_key])
            labels.append(self.gold_class)
        return [lut_idx_seqs, labels]

class STBParser:
    """
    Parse trees in the format of the train/dev/test set provided for the paper
    "Recursive Deep Models for Semantic Compositionality over a Sentiment
    Treebank" by Socher et. al, EMNLP 2013.

    TODO: More docs.
    """
    def __init__(self, tree_dir, min_freq=2, use_all_words=False):
        # Get file names for train/dev/test trees
        train_file = "{0:s}/train.txt".format(tree_dir)
        dev_file = "{0:s}/dev.txt".format(tree_dir)
        test_file = "{0:s}/test.txt".format(tree_dir)
        # Parse files for train/dev/test trees
        self.train_trees = self.parse_tree_file(train_file)
        self.dev_trees = self.parse_tree_file(dev_file)
        self.test_trees = self.parse_tree_file(test_file)
        # Get the set of all word occurrences in the training trees
        train_words = self.get_all_words(self.train_trees)
        if use_all_words:
            # Keep all words present in the corpus in the vocab
            train_words.extend(self.get_all_words(self.dev_trees))
            train_words.extend(self.get_all_words(self.test_trees))
            min_freq = 0
        # Compute maps from words to LUT keys and visa-versa, while discarding
        # words that occur fewer than 'min_freq' times.
        w2k, k2w = make_key_dicts(train_words, min_freq=min_freq, \
                                       unk_word='*UNK*')
        self.words_to_keys = w2k
        self.keys_to_words = k2w
        # Set the look-up-table indices for train/dev/test trees
        for tree in self.train_trees:
            tree.set_lut_keys(self.words_to_keys, unk_word='*UNK*')
        for tree in self.dev_trees:
            tree.set_lut_keys(self.words_to_keys, unk_word='*UNK*')
        for tree in self.test_trees:
            tree.set_lut_keys(self.words_to_keys, unk_word='*UNK*')
        return

    def parse_tree_file(self, f_name):
        """Parse an STB file."""
        root_tokens = [line.strip('\n') for line in open(f_name)]
        trees = [STBNode(token=rt, auto_spawn=True) for rt in root_tokens]
        return trees

    def get_all_words(self, tree_list):
        """Get the set of unique word tokens for the trees in "tree_list"."""
        all_words = []
        for tree in tree_list:
            all_words.extend(tree.get_all_tokens())
        return all_words



def LoadSTB(tree_dir, min_freq=2, use_all_words=False):
    """
    Load Stanford Treebank train/dev/test trees in a simple format.

    This converts all trees in the original train/validate/test files into
    lists of keys for indexing into a look-up-table. A map from LUT keys to
    words is returned in result['keys_to_words'] and the inverse map is
    returned in result['words_to_keys']. The full phrases and their labels
    (i.e. not including sub-phrase information) are returned in
    result['*_full_phrases'] and result['*_full_labels'] for * in {train, dev,
    test}. Similar information, but including all sub-phrases, is returned in
    result['*_phrases'] and result['*_labels'].

    If use_all_words is True, then all words from the train/dev/test sets will
    be included in the vocabulary/LUTs. If not, then LUT keys will only be
    assigned to words which appear at least min_freq times in full phrases from
    the training set. All other words will be assigned a LUT key associated
    with the word/token '*UNK*'.
    """
    # Parse the tree text files
    stbp = STBParser(tree_dir, min_freq=min_freq, use_all_words=use_all_words)
    dataset = {}
    # Get the vocab list (and look-up-table index map)
    dataset['words_to_keys'] = stbp.words_to_keys
    dataset['keys_to_words'] = stbp.keys_to_words
    # Get the training set phrases and labels
    dataset['train_phrases'] = []
    dataset['train_labels'] = []
    dataset['train_full_phrases'] = []
    dataset['train_full_labels'] = []
    for tree in stbp.train_trees:
        lutis_and_labels = tree.get_lutis_and_labels()
        dataset['train_full_phrases'].append(lutis_and_labels[0][-1])
        dataset['train_full_labels'].append(lutis_and_labels[1][-1])
        dataset['train_phrases'].extend(lutis_and_labels[0])
        dataset['train_labels'].extend(lutis_and_labels[1])
    # Get the development set phrases and labels
    dataset['dev_phrases'] = []
    dataset['dev_labels'] = []
    dataset['dev_full_phrases'] = []
    dataset['dev_full_labels'] = []
    for tree in stbp.dev_trees:
        lutis_and_labels = tree.get_lutis_and_labels()
        dataset['dev_full_phrases'].append(lutis_and_labels[0][-1])
        dataset['dev_full_labels'].append(lutis_and_labels[1][-1])
        dataset['dev_phrases'].extend(lutis_and_labels[0])
        dataset['dev_labels'].extend(lutis_and_labels[1])
    # Get the testing set phrases and labels
    dataset['test_phrases'] = []
    dataset['test_labels'] = []
    dataset['test_full_phrases'] = []
    dataset['test_full_labels'] = []
    for tree in stbp.test_trees:
        lutis_and_labels = tree.get_lutis_and_labels()
        dataset['test_full_phrases'].append(lutis_and_labels[0][-1])
        dataset['test_full_labels'].append(lutis_and_labels[1][-1])
        dataset['test_phrases'].extend(lutis_and_labels[0])
        dataset['test_labels'].extend(lutis_and_labels[1])
    # Convert to np.uint32 (which we always use for LUT keys)
    for set_str in ['train', 'dev', 'test']:
        s = '{0:s}_full_phrases'.format(set_str)
        dataset[s] = [np.array(p).astype(np.uint32) for p in dataset[s]]
        s = '{0:s}_phrases'.format(set_str)
        dataset[s] = [np.array(p).astype(np.uint32) for p in dataset[s]]
    return dataset

def parse_1bwords_file(f_name):
    """Parse the "1 Billion Words..." corpus file found @ f_name."""
    txt_phrases = [[w.lower() for w in l.strip().split()] for l in open(f_name)]
    txt_phrases = [p for p in txt_phrases if len(p) > 2]
    return txt_phrases

def Load1BWords(data_dir='./training_text', file_count=100, min_freq=5):
    import os
    # Get the list of relevant files in the given directory
    txt_files = [f for f in os.listdir(data_dir) if (f.find('news.en-') > -1)]
    if file_count > len(txt_files):
        file_count = len(txt_files)
    # Get all (already tokenized) sentences from the given files
    txt_phrases = []
    for i in range(file_count):
       txt_phrases.extend(parse_1bwords_file("{0:s}/{1:s}".format(data_dir, txt_files[i])))
    # Make dicts for words -> LUT keys and LUT keys -> words
    w2k, k2w = make_key_dicts(txt_phrases, min_freq=min_freq, unk_word='*UNK*')
    # Create LUT key representations of each phrase
    lk_phrases = [[(w2k[w] if (w in w2k) else w2k['*UNK*']) for w in p] for p in txt_phrases]
    lk_phrases = [np.asarray(p).astype(np.uint32) for p in lk_phrases]
    # Partition the dataset into training and validation parts
    split_idx = (4 * len(lk_phrases)) // 5
    dataset = {}
    dataset['words_to_keys'] = w2k
    dataset['keys_to_words'] = k2w
    dataset['train_key_phrases'] = lk_phrases[:split_idx]
    dataset['dev_key_phrases'] = lk_phrases[split_idx:]
    return dataset




###############################################################
# Basic testing, to see the functions aren't _totally_ broken #
###############################################################

if __name__ == '__main__':
    dataset = LoadSTB('./trees', min_freq=3, use_all_words=False)




##############
# EYE BUFFER #
##############

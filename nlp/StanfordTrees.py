import numpy as np
import numpy.random as npr

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
        self.lut_idx = -1
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

    def get_tokens(self):
        """Get a list of all tokens in the phrase rooted at this node."""
        tokens = []
        if self.has_children:
            tokens.extend(self.left_child.get_tokens())
            tokens.extend(self.right_child.get_tokens())
        else:
            assert (not (self.word is None))
            tokens = [self.word]
        return tokens

    def set_lut_idx(self, lut_keys):
        """Set numeric look-up-table indices for words in the subtree rooted
        at this node, based on the dict "lut_keys"."""
        if self.has_children:
            self.left_child.set_lut_idx(lut_keys)
            self.right_child.set_lut_idx(lut_keys)
        else:
            assert (not (self.word is None))
            if (self.word in lut_keys):
                # The word here is known by the dict
                self.lut_idx = lut_keys[self.word]
            else:
                # The word here is not known by the dict
                self.lut_idx = lut_keys["*UNK*"]
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
            assert (self.lut_idx >= 0)
            lut_idx_seqs.append([self.lut_idx])
            labels.append(self.gold_class)
        return [lut_idx_seqs, labels]

class STBParser:
    """
    Parse trees in the format of the train/dev/test set provided for the paper
    "Recursive Deep Models for Semantic Compositionality over a Sentiment
    Treebank" by Socher et. al, EMNLP 2013.

    TODO: More docs.
    """
    def __init__(self, tree_dir):
        # Get file names for train/dev/test trees
        train_file = "{0:s}/train.txt".format(tree_dir)
        dev_file = "{0:s}/dev.txt".format(tree_dir)
        test_file = "{0:s}/test.txt".format(tree_dir)
        # Parse files for train/dev/test trees
        self.train_trees = self.parse_tree_file(train_file)
        self.dev_trees = self.parse_tree_file(dev_file)
        self.test_trees = self.parse_tree_file(test_file)
        # Get a dict containing all tokens in the training trees
        train_words = self.get_vocab(self.train_trees)
        # Make a mapping from the training vocab to look-up-table indices
        self.lut_keys = {}
        self.lut_reverse = {}
        idx = 0
        for word in train_words:
            self.lut_keys[word] = idx
            self.lut_reverse[idx] = word
            idx = idx + 1
        self.lut_keys["*UNK*"] = idx
        # Set the look-up-table indices for train/dev/test trees
        for tree in self.train_trees:
            tree.set_lut_idx(self.lut_keys)
        for tree in self.dev_trees:
            tree.set_lut_idx(self.lut_keys)
        for tree in self.test_trees:
            tree.set_lut_idx(self.lut_keys)
        return

    def parse_tree_file(self, f_name):
        """Parse an STB file."""
        root_tokens = [line.strip('\n') for line in open(f_name)]
        trees = [STBNode(token=rt, auto_spawn=True) for rt in root_tokens]
        return trees

    def get_vocab(self, tree_list):
        """Get the set of unique word tokens for the trees in "tree_list"."""
        tl_vocab = []
        for tree in tree_list:
            tl_vocab.extend(tree.get_tokens())
        tl_vocab = set(tl_vocab)
        return tl_vocab

def SimpleLoad(tree_dir, keep_trees_grouped=True):
    """Load Stanford Treebank train/dev/test trees in a minimal format."""
    # Parse the tree text files
    stbp = STBParser(tree_dir)
    dataset = {}
    dataset['trees_are_grouped'] = keep_trees_grouped
    # Get the vocab list (and look-up-table index map)
    dataset['lut_keys'] = stbp.lut_keys
    # Get the training set phrases and labels
    dataset['train_phrases'] = []
    dataset['train_labels'] = []
    dataset['train_full_phrases'] = []
    dataset['train_full_labels'] = []
    for tree in stbp.train_trees:
        lutis_and_labels = tree.get_lutis_and_labels()
        dataset['train_full_phrases'].append(lutis_and_labels[0][-1])
        dataset['train_full_labels'].append(lutis_and_labels[1][-1])
        if keep_trees_grouped:
            dataset['train_phrases'].append(lutis_and_labels[0])
            dataset['train_labels'].append(lutis_and_labels[1])
        else:
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
        if keep_trees_grouped:
            dataset['dev_phrases'].append(lutis_and_labels[0])
            dataset['dev_labels'].append(lutis_and_labels[1])
        else:
            dataset['dev_phrases'].extend(lutis_and_labels[0])
            dataset['dev_labels'].extend(lutis_and_labels[1])
    # Get the testing set phrases and labels
    dataset['test_phrases'] = []
    dataset['test_labels'] = []
    dataset['test_full_phrases'] = []
    dataset['test_full_labels'] = []
    for tree in stbp.train_trees:
        lutis_and_labels = tree.get_lutis_and_labels()
        dataset['test_full_phrases'].append(lutis_and_labels[0][-1])
        dataset['test_full_labels'].append(lutis_and_labels[1][-1])
        if keep_trees_grouped:
            dataset['test_phrases'].append(lutis_and_labels[0])
            dataset['test_labels'].append(lutis_and_labels[1])
        else:
            dataset['test_phrases'].extend(lutis_and_labels[0])
            dataset['test_labels'].extend(lutis_and_labels[1])
    # Intentionally forget the memory hog 
    del stbp
    return dataset

###############################################################
# Basic testing, to see the functions aren't _totally_ broken #
###############################################################

if __name__ == '__main__':
    stb_parser = STBParser('./trees/dev.txt')




##############
# EYE BUFFER #
##############

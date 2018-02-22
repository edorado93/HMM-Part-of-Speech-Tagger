import os.path

""" The most basic element of the data. A (word, tag) tuple """
class Atom:
    delimiter = "/"
    def __init__(self, data, is_training):
        self.word = None
        self.tag = None
        self._populate(data, is_training)

    def _populate(self, data, is_training):
        if self.delimiter not in data and is_training:
            print("No tag found in training data", data)
        elif is_training:
            split = data.split(self.delimiter)
            self.word = "/".join(split[:-1])
            self.tag = split[-1]
        else:
            self.word = data

    def get_tag(self):
        return self.tag

    def get_word(self):
        return self.word

class DataParser:
    def __init__(self, corpus_files):
        """ Filename variables """
        self.train_tagged, is_train_1 = corpus_files[0]
        self.test_tagged, is_train_2 = corpus_files[1]
        self.test_raw, is_train_3 = corpus_files[2]

        """
            Lists to actually hold the sentences. Each
            sentence is a collection of tuples of the form(word, tag)
        """
        self.train_sentences = []

        """ 
            Gold sentences from the test corpus to compare against.
            Similarly we have the raw sentences which are untagged. These 
            would be tagged by our HMM and then we will calculate 
            the accuracy. 
        """
        self.gold_sentences = []
        self.raw_sentences = []

        self._populate_data(is_train_1, is_train_2, is_train_3)

    def get_raw_test(self):
        return self.raw_sentences

    def get_training_data(self):
        return self.train_sentences

    def get_gold_standard(self):
        return self.gold_sentences

    """ Parse all the files and populate all the data structures. """
    def _populate_data(self, is_train_1, is_train_2, is_train_3):
        self._parse_file(self.train_tagged, self.train_sentences, is_training=is_train_1)
        self._parse_file(self.test_tagged, self.gold_sentences, is_training=is_train_2)
        self._parse_file(self.test_raw, self.raw_sentences, is_training=is_train_3)

    def _parse_file(self, filename, lst, is_training):
        if not filename or not os.path.isfile(filename):
            raise Exception("Invalid file")

        with open(filename, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Split the sentence into word_tag pieces. Considering space as the delimiter
                word_tag = line.split()

                # Convert string word/tag to Atom object
                word_tag_atomised = [Atom(data, is_training) for data in word_tag]

                # Add the atomised sentence to our list
                lst.append(word_tag_atomised)

    def get_lists(self):
        return (self.train_tagged, self.test_tagged, self.test_raw)

class KFoldCrossValidation:
    def __init__(self, k, training_data):
        self.k = k if k > 0 else 1 # Edge case where we want the entire data for training. Not much of split.
        self.data = training_data
        self.portions = []
        self.split()

    """ We split the entire corpus into equal portions """
    def split(self):
        num_of_sentences_per_split = int(len(self.data) / self.k)
        split = []
        for i, sentence in enumerate(self.data):
            split.append(sentence)
            if (i+1) % num_of_sentences_per_split == 0:
                self.portions.append(split)
                split = []

        if split:
            self.portions.append(split)

    def get_train_and_test_data(self, index=0):
        return self.portions[index], [portion for i, portion in enumerate(self.portions) if i != index], index+1
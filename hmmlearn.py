from data import *
import _pickle as pickle
import math
import sys

class ConditionalProbability:
    # a fake START tag to add to the beginning of sentences to help with tagging
    start_tag = '^'

    # x-fold cross-validation
    k_fold = 10

    def __init__(self, corpus_files):
        # will hold conditional frequency distribution for P(Wi|Ck)
        self.words_given_pos = {}

        # will hold conditional frequency distribution for P(Ci+2|Ci+1, Ci)
        self.pos3_given_pos2_and_pos1 = {}

        # A helper object that gives us access to parsed files' data, both test and train.
        self.parser = DataParser(corpus_files)

        # An iterator over KFoldCrossValidation logic.
        self.cycle_iterator = KFoldCrossValidation(ConditionalProbability.k_fold, self.parser.get_training_data())

        # Mapping of a word to set of tags it occurred with in the entire corpus
        self.word_to_tag = {}

        # Count of word and tag occurring together.
        self.word_tag_count = {}

        # Count of given word in the entire corpus
        self.word_count = {}

        # Count the occurrence of 3 tags in a sequence
        self.trigram_counts = {}

        # Count of two tags (prior) in a sequence
        self.bigram_counts = {}

        # Set of all the tags in the corpus
        self.tags = set()

    def calculate_probabilities(self):
        self.populate_dictionaries()
        self.CFDWordGivenTag()
        self.CFDTrigramTags()
        self._save()

    def _save(self):
        dictionaries = {"unique_tags" : self.tags, "bigram" : self.bigram_counts, "transmission" : self.pos3_given_pos2_and_pos1, "emission" : self.words_given_pos, "word2tag" : self.word_to_tag}
        output = open('hmmmodel.txt', 'wb')
        pickle.dump(dictionaries, output)
        output.close()

    def populate_dictionaries(self):
        self.pos_tags = set()
        for sentence in self.parser.get_training_data():
            """ Populating the tag dictionaries. Required for TRANSITION probabilities """

            """ Added two start symbols for the trigram consideration from the first word """
            sentence.insert(0, Atom(ConditionalProbability.start_tag+Atom.delimiter+ConditionalProbability.start_tag, is_training=True))
            sentence.insert(0, Atom(ConditionalProbability.start_tag+Atom.delimiter+ConditionalProbability.start_tag, is_training=True))

            """ The first word in the original sentence """
            start_index = 2
            for ind in range(start_index, len(sentence)):
                trigram_triplet = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag(), (sentence[ind]).get_tag())
                bigram_tuple = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag())
                self.trigram_counts[trigram_triplet] = self.trigram_counts.get(trigram_triplet, 0) + 1
                self.bigram_counts[bigram_tuple] = self.bigram_counts.get(bigram_tuple, 0) + 1

            """ Populating the word dictionaries. Required for EMISSION probabilities """
            for i, atom in enumerate(sentence):

                word = atom.get_word()
                tag = atom.get_tag()

                self.tags.add(tag)

                self.word_tag_count[ (word, tag) ] =  self.word_tag_count.get((word, tag), 0) + 1
                self.word_count[ word ] = self.word_count.get(word, 0) + 1
                if word not in self.word_to_tag:
                    self.word_to_tag[ word ] = set()
                self.word_to_tag[ word ].add(tag)

    """
        Probability of a word given a tag.
        C(s -> x) is the count of word "s" and tag "x" occurring together.
        C(s) is the count of word "s" occurring in the corpus.
        E(x|s) = C(s -> x) / C(s)
    """
    def CFDWordGivenTag(self):
        for word, tags_set in self.word_to_tag.items():
            for tag in tags_set:
                self.words_given_pos[(word, tag)] = math.log(float(self.word_tag_count[(word, tag)]) / float(self.word_count[word]))

    """  
        Probability of a tag "s" given previous 2 tags as u and v
        C(u, v, s) is the count of this triplet occurring together in this order in the corpus
        C(u, v) is the count of this tuple occurring together in this order in the corpus
        P(s|u, v) = C(u, v, s) / C(u, v) 
    """
    def CFDTrigramTags(self):
        """ The first word in the original sentence """
        start_index = 2
        V = len(self.tags)
        for sentence in self.parser.get_training_data():
            for ind in range(start_index, len(sentence)):
                trigram_triplet = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag(), (sentence[ind]).get_tag())
                bigram_tuple = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag())
                self.pos3_given_pos2_and_pos1[trigram_triplet] = math.log(float(1 + self.trigram_counts[trigram_triplet]) /\
                                                                 float(V + self.bigram_counts[bigram_tuple]))


if __name__ == "__main__":

    # conf_prob = ConditionalProbability(["coding1-data-corpus/homework2_tagged.txt",
    #                                     "coding1-data-corpus/zh_dev_tagged.txt",
    #                                     "coding1-data-corpus/zh_dev_raw.txt"])
    # conf_prob.calculate_probabilities()

    filename = sys.argv[1]
    conf_prob = ConditionalProbability([(filename, True), (filename, True), (filename, True)])
    conf_prob.calculate_probabilities()
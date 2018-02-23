from data import *
import _pickle as pickle
import math
import sys

class ConditionalProbability:
    # a fake START tag to add to the beginning of sentences to help with tagging
    start_tag = '$^START^$'

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
        self.tag_count = {}

        # Count the occurrence of 3 tags in a sequence
        self.trigram_counts = {}

        # Count of two tags (prior) in a sequence
        self.bigram_counts = {}

        # Set of all the tags in the corpus
        self.tags = set()

        # Set of all the words in the corpus
        self.words = set()

        """ Back-off Probabilities """
        self.transition_backoff = {}
        self.emission_backoff = {}

        """ Singleton counts """
        self.transition_singleton = {}
        self.emission_singleton = {}

        """ 1-count smoothed probabilities """
        self.transition_one_count = {}
        self.emission_smoothed = {}

        self.n = 0

    def calculate_probabilities(self):
        self.populate_dictionaries()
        self.CFDWordGivenTag()
        self.CFDTrigramTags()
        self.BackoffProbabilities()
        self.SingletonCounts()
        self.SmoothedProbabilities()
        self._save()

    def _save(self):
        dictionaries = {"unique_tags" : self.tags, "bigram" : self.bigram_counts, "transmission" : self.pos3_given_pos2_and_pos1, "emission" : self.words_given_pos, "word2tag" : self.word_to_tag,
                        "transition_backoff" : self.transition_backoff, "emission_backoff" : self.emission_backoff,
                        "transition_singleton" : self.transition_singleton, "emission_singleton" : self.emission_singleton,
                        "transition_smoothed" : self.transition_one_count, "emission_smoothed" : self.emission_smoothed,
                        "tag_count" : self.tag_count, "n" : self.n}
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
                self.n += 1

                """ Backoff counts """
                self.transition_backoff[tag] = self.transition_backoff.get(tag, 0) + 1
                self.emission_backoff[word] = self.emission_backoff.get(word, 0) + 1

                self.tags.add(tag)
                self.words.add(word)

                self.word_tag_count[ (word, tag) ] =  self.word_tag_count.get((word, tag), 0) + 1
                self.tag_count[ tag ] = self.tag_count.get(tag, 0) + 1
                if word not in self.word_to_tag:
                    self.word_to_tag[ word ] = set()
                self.word_to_tag[ word ].add(tag)

        print(self.trigram_counts)
        print("*"*50)
        print(self.bigram_counts)

    def BackoffProbabilities(self):
        V = len(self.tags)
        print(self.n, V)
        for word in self.emission_backoff:
            self.emission_backoff[word] = float(1 + self.emission_backoff[word]) / float(self.n + V)

        for tag in self.transition_backoff:
            self.transition_backoff[tag] = float(self.transition_backoff[tag]) / float(self.n)

    def SingletonCounts(self):
        for i, tag_1 in enumerate(self.tags):
            for j, tag_2 in enumerate(self.tags):
                for k, tag_3 in enumerate(self.tags):
                    if i != j and i != k and j != k:
                        triplet = (tag_3, tag_2, tag_1)
                        if triplet in self.trigram_counts and self.trigram_counts[triplet] == 1:
                            self.transition_singleton[(tag_3, tag_2)] = self.transition_singleton.get((tag_3, tag_2), 0) + 1

        for word in self.words:
            for tag in self.tags:
                word_tag = (word, tag)
                if word_tag in self.word_tag_count and self.word_tag_count[word_tag] == 1:
                    self.emission_singleton[tag] = self.emission_singleton.get(tag, 0) + 1

    def SmoothedProbabilities(self):
        start_index = 2
        for sentence in self.parser.get_training_data():
            for ind in range(start_index, len(sentence)):
                trigram_triplet = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag(), (sentence[ind]).get_tag())
                bigram_tuple = ((sentence[ind - 2]).get_tag(), (sentence[ind - 1]).get_tag())
                lamda = 1 + self.transition_singleton.get(bigram_tuple, 0)
                self.transition_one_count[trigram_triplet] = math.log(float(self.trigram_counts[trigram_triplet] + lamda * self.transition_backoff[sentence[ind].get_tag()]) /\
                float(self.bigram_counts[bigram_tuple] + lamda))

        for word, tags_set in self.word_to_tag.items():
            for tag in tags_set:
                lamda = 1 + self.emission_singleton.get(tag, 0)
                self.emission_smoothed[(word, tag)] = math.log(float(self.word_tag_count[(word, tag)] + lamda * self.emission_backoff[word]) /\
                                                               float(self.tag_count[tag] + lamda))

    """
        Probability of a word given a tag.
        C(s -> x) is the count of word "s" and tag "x" occurring together.
        C(s) is the count of word "s" occurring in the corpus.
        E(x|s) = C(s -> x) / C(s)
    """
    def CFDWordGivenTag(self):
        for word, tags_set in self.word_to_tag.items():
            for tag in tags_set:
                self.words_given_pos[(word, tag)] = math.log(float(self.word_tag_count[(word, tag)]) / float(self.tag_count[tag]))

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
import _pickle as pickle
from data import *
from hmmlearn import *
import sys

class HMM:

    transition_minimum = -100000

    def __init__(self, corpus_files, saved_model):

        """ Transmission probabilities saved by running the tranining """
        self.transition = None

        """ Emission probabilities. Probabilities of tags given words. """
        self.emission = None

        """ Helper file to give us access to the training corpus """
        self.data_helper = DataParser(corpus_files)

        """ File containing the saved model parameters """
        self.model = saved_model

        """ Given a word, get all it's tags from the training corpus """
        self.word2tag = None

        self.backpointers = {}

        self.viterbi_probabilities = {}

        self.bigram_counts = {}

        self.unique_tags = set()

        """ Back-off Probabilities """
        self.transition_backoff = {}
        self.emission_backoff = {}

        """ Singleton counts """
        self.transition_singleton = {}
        self.emission_singleton = {}

        """ 1-count smoothed probabilities """
        self.transition_one_count = {}
        self.emission_smoothed = {}

        self.tag_count = {}
        self.n = 0

    def load(self):
        f = open(self.model, "rb")
        dictionaries = pickle.load(f)
        self.transition = dictionaries["transmission"]
        self.emission = dictionaries["emission"]
        self.word2tag = dictionaries["word2tag"]
        self.bigram_counts = dictionaries["bigram"]
        self.unique_tags = dictionaries["unique_tags"]

        """ New probabilities """
        self.transition_backoff = dictionaries["transition_backoff"]
        self.emission_backoff = dictionaries["emission_backoff"]
        self.transition_singleton = dictionaries["transition_singleton"]
        self.emission_singleton = dictionaries["emission_singleton"]
        self.transition_one_count = dictionaries["transition_smoothed"]
        self.emission_smoothed = dictionaries["emission_smoothed"]
        self.tag_count = dictionaries["tag_count"]
        self.n = dictionaries["n"]

    def base_case(self, word, current_tag):
        emission = self._get_smoothed_emission(word, current_tag)
        transition = self._get_smoothed_transition(ConditionalProbability.start_tag, ConditionalProbability.start_tag, current_tag)
        return emission + transition, transition

    def _recover_tags(self, sentence):
        pos_tag_indices = []
        for j, word in reversed(list(enumerate(sentence))):
            if j == len(sentence) - 1:
                maxi = HMM.transition_minimum
                insert_value = None
                for tag1 in self._get_tags(word):
                    for tag2 in self._get_tags(sentence[j - 1]):
                        tag_tuple = (tag2, tag1)
                        if self.viterbi_probabilities[(tag_tuple, j)] > maxi:
                            maxi = self.viterbi_probabilities[(tag_tuple, j)]
                            insert_value = (word, tag_tuple)
                pos_tag_indices.insert(0, insert_value)
            else:
                pos_tag_indices.insert(0, (word, self.backpointers[(pos_tag_indices[0][1], j + 1)]))
        return [(tup[0], tup[1][1]) for tup in pos_tag_indices[1:]]

    def _get_bigram_counts(self, u, v):
        if (u, v) in self.bigram_counts:
            return self.bigram_counts[(u, v)]
        return 0

    def _get_tags(self, word):
        if word in self.word2tag:
            return self.word2tag[word]
        return self.unique_tags

    def _get_emission_backoff(self, word):
        if word in self.emission_backoff:
            return self.emission_backoff[word]

        V = len(self.unique_tags)
        return float(1) / float(self.n + V)

    def _get_smoothed_emission(self, word, tag):
        if (word, tag) in self.emission_smoothed:
            return self.emission_smoothed[(word, tag)]
        else:
            lamda = 1 + self.emission_singleton.get(tag, 0)
            return math.log(float(lamda * self._get_emission_backoff(word)) /\
                                                               float(self.tag_count[tag] + lamda))

    def _get_smoothed_transition(self, tag_k, tag_j, tag_i):
        if (tag_k, tag_j, tag_i) in self.transition_one_count:
            """ transition probability P(tag_i | tag_i - 1, tag_i - 2) """
            transition_probability = self.transition_one_count[(tag_k, tag_j, tag_i)]
        else:
            """ Smoothed transition probability """
            bigram_counts = self._get_bigram_counts(tag_k, tag_j)
            lamda = 1 + self.transition_singleton.get((tag_k, tag_j), 0)
            transition_probability = math.log(float(lamda * self.transition_backoff[tag_i]) /\
                float(bigram_counts + lamda))
        return transition_probability

    def decode(self, sentence):
       """ loop through words """
       sentence.insert(0, ConditionalProbability.start_tag)
       for j in range(1, len(sentence)):
           word = sentence[j]
           """ loop through possible POS tags """
           for tag_i in self._get_tags(word):

                """ Base case needs to be handled here """
                if j == 1:
                    tag_tuple = (ConditionalProbability.start_tag, tag_i)
                    viterbi, transition = self.base_case(word, tag_i)

                    """ # calculate score using P(Ci|'^', '^') and P(Wj|Ci) """
                    self.viterbi_probabilities[(tag_tuple, j)] = viterbi

                    """ initialize backpointer for this word to 0 """
                    self.backpointers[(tag_tuple, j)] = (ConditionalProbability.start_tag, ConditionalProbability.start_tag)

                else:
                    max_viterbi = HMM.transition_minimum
                    backpointer = None

                    """ Emission probability P(w | tag_i) """
                    emission_probability = self._get_smoothed_emission(word, tag_i)

                    """ Loop over all possible pair of tags for the previous 2 words """
                    for tag_j in self._get_tags(sentence[j - 1]):
                        tag_tuple = (tag_j, tag_i)  # TODO can be reduced to just the previous tag
                        """ All possible tags for current - 2th word """
                        for tag_k in self._get_tags(sentence[j - 2]):
                            transition_probability = self._get_smoothed_transition(tag_k, tag_j, tag_i)

                            """ Viterbi Log probability """
                            viterbi = self.viterbi_probabilities[((tag_k, tag_j), j - 1)] + transition_probability + emission_probability

                            """ Calculating the max and backpointer to recover the tag sequence """
                            if viterbi > max_viterbi:
                                max_viterbi = viterbi
                                backpointer = (tag_k, tag_j)

                        self.viterbi_probabilities[(tag_tuple, j)] = max_viterbi
                        self.backpointers[(tag_tuple, j)] = backpointer

       pos_tagging = self._recover_tags(sentence)
       # print("Printing POS tags")
       # for element in pos_tagging:
       #     print(element)

       word_tag = []
       for word, tag in pos_tagging:
            word_tag.append(Atom(word+Atom.delimiter+tag, True))

       return word_tag


    def indices_of_max(self, arr):
        """
        Return the index/indices in an array which have the highest value.
        :param array: list to search
        """

        indices = []  # intialize index list

        # for each item in array, append index if it has max value
        maxi = max(arr, key=lambda x: x[1])[1]
        for i in range(len(arr)):
            if arr[i][1] == maxi:
                indices.append(arr[i][0])

        return indices

    def match(self, hmm_tagged_sents, gold_tagged_sents):
        """
        Evaluate one set of tagged sentences against another set

        :param hmm_tagged_sents: list of tagged sentences
        :param gold_tagged_sents: list of tagged sentences used as gold standard
        """

        # ensure our sentence sets have the same length
        if len(hmm_tagged_sents) != len(gold_tagged_sents):
            raise Exception("HMM-tagged sentence set did not match gold \
                standard sentence set!")

        right = 0  # initialize counter of correct tags
        wrong = 0  # initialize counter of incorrect tags

        # loop through sentence sets
        for i in range(len(gold_tagged_sents)):

            # ensure our sentences have the same length
            if len(hmm_tagged_sents[i]) != len(gold_tagged_sents[i]):
                raise Exception("HMM-tagged sentence did not match gold \
                    standard sentence!", len(hmm_tagged_sents), len(gold_tagged_sents))

            # loop through words in sentence
            for j in range(len(gold_tagged_sents[i])):
                gold_tagged_word = gold_tagged_sents[i][j]
                hmm_tagged_word = hmm_tagged_sents[i][j]

                # ensure the words are the same between the sets
                if gold_tagged_word.get_word() != hmm_tagged_word.get_word():
                    raise Exception("HMM-tagged word did not match gold \
                        standard word!")

                # increment counters based on tag correctness
                if gold_tagged_word.get_tag() == hmm_tagged_word.get_tag():
                    right += 1
                else:
                    # missed.append((hmm_tagged_word, gold_tagged_word, \
                    #                hmm_tagged_sents[i], gold_tagged_sents[i]))
                    wrong += 1

        return (right, wrong)

    def evaluate(self):
        generated_tags = []
        gold_tags = []
        with open(self.data_helper.test_raw, "r", encoding='utf-8') as untagged_sentences:
            for line in untagged_sentences:
                l = []
                for word in line.split():
                    l.append(Atom(word, True))
                generated_tags.append(l)

        with open(self.data_helper.test_tagged, "r", encoding='utf-8') as tagged_development:
            for line in tagged_development:
                l = []
                for word in line.split():
                    l.append(Atom(word, True))
                gold_tags.append(l)

        right, wrong = self.match(generated_tags, gold_tags)
        return right, wrong

    def run(self):
        output = open("hmmoutput.txt", "w")
        untagged_sentences = self.data_helper.get_raw_test()
        for sentence in untagged_sentences:
            sentence = [x.get_word() for x in sentence]
            pos_tags = self.decode(sentence)
            ans = []
            for word, atom in zip(sentence[1:], pos_tags):
                ans.append(word+"/"+atom.get_tag())
            output.write(" ".join(ans) + "\n")

        output.close()

if __name__ == "__main__":
    # hmm = HMM([("coding1-data-corpus/en_train_tagged.txt", True),
    #            ("coding1-data-corpus/en_dev_tagged.txt", True),
    #            ("hmmoutput2.txt", True)], "hmmmodel.txt")
    # hmm.load()
    # r, w = hmm.evaluate()
    # print((float(r) * 100)/ float(w+r))

    filename = sys.argv[1]
    hmm = HMM([(filename, False), (filename, False), (filename, False)], "hmmmodel.txt")
    hmm.load()
    hmm.run()
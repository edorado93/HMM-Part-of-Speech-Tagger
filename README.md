# Overview
In this assignment you will write a Hidden Markov Model part-of-speech tagger for English, Chinese, and a surprise language. The training data are provided tokenized and tagged. The test data will be provided tokenized, and your tagger will add the tags. 

# Data
The data would consist of:

* Two files (one English, one Chinese) with tagged training data in the word/TAG format, with words separated by spaces and each sentence on a new line.

* Two files (one English, one Chinese) with untagged development data, with words separated by spaces and each sentence on a new line.

* Two files (one English, one Chinese) with tagged development data in the word/TAG format, with words separated by spaces and each sentence on a new line, to serve as an answer key.

# Programs
You will write two programs: `hmmlearn.py` will learn a hidden Markov model from the training data, and `hmmdecode.py` will use the model to tag new data. The learning program will be invoked in the following way:

``` > python hmmlearn.py /path/to/input ```

The argument is a single file containing the training data; the program will learn a hidden Markov model, and write the model parameters to a file called `hmmmodel.txt`. The format of the model is up to you, but it should follow the following guidelines:

* The model file should contain sufficient information for `hmmdecode.py` to successfully tag new data.

* The model file should be human-readable, so that model parameters can be easily understood by visual inspection of the file.

The tagging program will be invoked in the following way:

```> python hmmdecode.py /path/to/input```

The argument is a single file containing the test data; the program will read the parameters of a hidden Markov model from the file `hmmmodel.txt`, tag each word in the test data, and write the results to a text file called `hmmoutput.txt` in the same format as the training data.

# Notes
### Tags 
Each language has a different tagset; the surprise language will have some tags that do not exist in the English and Chinese data. You must therefore build your tag sets from the training data, and not rely on a precompiled list of tags.

### Slash character
The slash character ‘/’ is the separator between words and tags, but it also appears within words in the text, so be very careful when separating words from tags. Slashes never appear in the tags, so the separator is always the last slash in the word/tag sequence.
Smoothing and unseen words and transitions. You should implement some method to handle unknown vocabulary and unseen transitions in the test data, otherwise your programs won’t work.

### Unseen words
The test data may contain words that have never been encountered in the training data: these will have an emission probability of zero for all tags.

### Unseen transitions
The test data may contain two adjacent unambiguous words (that is, words that can only have one part-of-speech tag), but the transition between these tags was never seen in the training data, so it has a probability of zero; in this case the Viterbi algorithm will have no way to proceed.

### End State
You may choose to implement the algorithm with transitions to an end state at the end of a sentence , or without an end state 

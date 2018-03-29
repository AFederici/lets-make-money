
# coding: utf-8

# # Train Your Own Word Embeddings
# 
# In this notebook, you'll create your own word embeddings by analyzing a corpus to see what words appears in what contexts.
# 
# The strategy is to:
# 
# - create a matrix X of counts of words appearing within a certain distance of other words (in a context window)
# - apply truncated SVD to reduce the dimensionality of the matrix down to the final dimension of the embedding space (e.g., 50, 100, 200, or 300)
# 
# The rows of matrix X are the words in the center of a context window, e.g., in
# 
#     i am scared of dogs
# 
# the word "scared" is in the center of a context of 2 words on either side. We want to store the count of the number of times a word like "dogs" appeared in the context of "scared". So we would update the element in the row corresponding to "scared" and the column corresponding to "dogs".
# 
# Because this matrix can grow very large as the vocabulary size grows, we're going to restrict the size of the vocabulary to the most frequenct `max_vocab_words` words. And we're going to restrict the words that we'll analyze as context words to the most frequent `max_context_words` words.
# 

# ## 0 Imports

# In[239]:

import numpy as np
from sklearn.decomposition import TruncatedSVD, randomized_svd
from gensim.models.keyedvectors import KeyedVectors
import time
from collections import defaultdict, Counter
import codecs
from nltk.tokenize import word_tokenize


# ## 2 Functions to create word by context matrix

# ### 2.1 Complete the following functions for converting tokens to word codes.

# In[104]:

def generate_sorted_words(tokens):
    """ Create list of unique words sorted by count in descending order
        
        Parameters
        ----------
        tokens: list(str)
            A list of tokens (words), e.g., ["the", "cat", "in", "the", "in", "the"]
        
        Returns
        -------
        list(str)
            A list of unique tokens sorted in descending order, e.g., ["the", "in", cat"]
        
    """
    # SOLUTION
    counter = Counter(tokens)
    words = [word for word, count in counter.most_common()]
    return words


# In[105]:

def generate_word2code(sorted_words):
    """ Create dict that maps a word to its position in the sorted list of words
    
        Parameters
        ---------
        sorted_words: list(str)
            A list of unique words, e.g., ["b", "c", "a"]
        
        Returns
        -------
        dict[str, int]
            A dictionary that maps a word to an integer code, e.g., {"b": 0, "c": 1, "a": 2}
        
    """
    # SOLUTION
    word2code = {w : i for i, w in enumerate(sorted_words)}
    return word2code


# In[106]:

def convert_tokens_to_codes(tokens, word2code):
    """ Convert tokens to codes.
    
        Parameters
        ---------
        tokens: list(str)
            A list of words, e.g., ["b", "c", "a"]
        word2code: dict[str, int]
            A dictionary mapping words to integer codes, e.g., {"b": 0, "c": 1, "a": 2}
        
        Returns
        -------
        list(int)
            A list of codes corresponding to the input words, e.g., [0, 1, 2].
    """
    # SOLUTION
    return [word2code[token] for token in tokens]


# ### 2.2 Test tokens to codes

# In[107]:

tokens = ["a", "a", "b", "c", "c", "c", "c", "a", "b", "c"]


# Value of `sorted_words` should be: `['c', 'a', 'b']`

# In[108]:

sorted_words = generate_sorted_words(tokens)
print(sorted_words)


# Value of `word2code` should be: `{'c': 0, 'a': 1, 'b': 2}`

# In[109]:

word2code = generate_word2code(sorted_words)
print(word2code)


# Value of `codes` should be: `[1, 1, 2, 0, 0, 0, 0, 1, 2, 0]`

# In[110]:

codes = convert_tokens_to_codes(tokens, word2code)
print(codes)


# ### 2.3 Complete the following fuction for creating a word by context matrix given a sequence of word codes.

# In[278]:

from numba import njit


# In[279]:

@njit
def generate_word_by_context(codes, max_vocab_words=1000, max_context_words=1000, context_size=2, weight_by_distance=False):
    """ Create matrix of vocab word by context word (possibly weighted) co-occurrence counts.
    
        Parameters
        ----------
        codes: list(int)
            A sequence of word codes.
        max_vocab_words: int
            The max number of words to include in vocabulary (will correspond to rows in matrix).
            This is equivalent to the max word code that will be considered/processed as the center word in a window.
        max_context_words: int
            The max number of words to consider as possible context words (will correspond to columns in matrix).
            This is equivalent to the max word code that will be considered/processed when scanning over contexts.
        context_size: int
            The number of words to consider on both sides (i.e., to the left and to the right) of the center word in a window.
        weight_by_distance: bool
            Whether or not the contribution of seeing a context word near a center word should be 
            (down-)weighted by their distance:
            
                False --> contribution is 1.0
                True  --> contribution is 1.0 / (distance between center word position and context word position)
            
            For example, suppose ["i", "am", "scared", "of", "dogs"] has codes [45, 10, 222, 25, 88]. 
            
            With weighting False, 
                X[222, 25], X[222, 10], X[222, 25], and X[222, 88] all get incremented by 1.
            
            With weighting True, 
                X[222, 25] += 1.0/2 
                X[222, 10] += 1.0/1 
                X[222, 25] += 1.0/1
                X[222, 88] += 1.0/2
        
        Returns
        -------
        (max_vocab_words x max_context_words) ndarray
            A matrix where rows are vocab words, columns are context words, and values are
            (possibly weighted) co-occurrence counts.
    """
    
    """
    pseudo-code:
    
    slide window along sequence
    if code of center word is < max_vocab_words
        for each word in context (on left and right sides)
            if code of context word < max_context_words
                add 1.0 to matrix element in row of center word and column of context word
                    or
                add 1.0 / (distance from center to context)
    
    example: assume context_size is 2 (i.e., 2 words to left and 2 words to right)
    
      "a" "a" "b" "c" "c" "c" "c" "a" "b" "c"   # sequence of words
       1   1   2   0   0   0   0   1   2   0    # sequence of word codes
       0   1   2   3   4   5   6   7   8   9    # position in sequence
      [        ^        ]                       # first window: centered on position 2; center word has code 2
          [        ^        ]                   # second window: centered on position 3; center word has code 0
                       ...                 
                          [        ^        ]   # last window: centered on position 7; center word has code 1
    """
    
    # initialize matrix (with dtype="float32" to reduce required memory)
    
    # SOLUTION
    X = np.zeros((max_vocab_words, max_context_words))

    # slide window along sequence and count "center word code" / "context word code" co-occurrences
    # Hint: let main loop index indicate the center of the window
    
    # SOLUTION
    for i in range(context_size, len(codes) - context_size):
#         if i % 100000 == 0:
#             print("i = " + str(i) + ": " + str(1.0 * i / len(codes)) + "%")

        center_code = codes[i]
        if center_code < max_vocab_words:
            # left side
            for j in range(1, context_size + 1):
                context_code = codes[i - j]
                if context_code < max_context_words:
                    value = 1.0
                    if weight_by_distance:
                        value = 1.0 / j
                    X[center_code, context_code] += value
            # right side
            for j in range(1, context_size + 1):
                context_code = codes[i + j]
                if context_code < max_context_words:
                    value = 1.0
                    if weight_by_distance:
                        value = 1.0 / j
                    X[center_code, context_code] += value

    return X


# ### 2.4 Test on sample sequences

# In[280]:

# assume this sequence already contains word codes
sequence = [2, 3, 4, 5, 6, 7]
X = generate_word_by_context(sequence, max_vocab_words=8, max_context_words=8, context_size=2, weight_by_distance=False)
print(X)


# Value of X should be:
# ````python
# [[ 0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  1.  1.  0.  1.  1.  0.]
#  [ 0.  0.  0.  1.  1.  0.  1.  1.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.]]
# ````
# since there are only two full windows as we slide along the sequence: [2, 3, 4, 5, 6] and [3, 4, 5, 6, 7].
# 
# In the first window, the code of the center word is 4, and it co-occurs with codes 2 and 3 on its left and codes 5 and 6 on its right.

# In[281]:

# turn on weighting
X = generate_word_by_context(sequence, max_vocab_words=8, max_context_words=8, context_size=2, weight_by_distance=True)
print(X)


# Value of X should now be:
# ````python
# [[ 0.   0.   0.   0.   0.   0.   0.   0. ]
#  [ 0.   0.   0.   0.   0.   0.   0.   0. ]
#  [ 0.   0.   0.   0.   0.   0.   0.   0. ]
#  [ 0.   0.   0.   0.   0.   0.   0.   0. ]
#  [ 0.   0.   0.5  1.   0.   1.   0.5  0. ]
#  [ 0.   0.   0.   0.5  1.   0.   1.   0.5]
#  [ 0.   0.   0.   0.   0.   0.   0.   0. ]
#  [ 0.   0.   0.   0.   0.   0.   0.   0. ]]
# ````
# There are still only two full windows: [2, 3, 4, 5, 6] and [3, 4, 5, 6, 7]. But now context words that are farther away from the center word contribute a lower weight.

# ### 3 Train on Wikipedia

# ### 3.1 Load Wikipedia file (update path if necessary)

# In[282]:

path_to_wikipedia = "../day1/wikipedia2text-extracted.txt"
with open(path_to_wikipedia, "rb") as f:
    wikipedia = f.read().decode().lower()
print(str(len(wikipedia)) + " character(s)")


# ### 2.2 Tokenize

# In[283]:

# we're going to use NLTK's tokenizer again, but leave in punctuation (which GloVe does as well)
tokens = word_tokenize(wikipedia.lower())
print(str(len(tokens)) + " tokens(s)")


# ### 2.3 Convert tokens to codes

# In[284]:

# SOLUTION
sorted_words = generate_sorted_words(tokens)
word2code = generate_word2code(sorted_words)
codes = convert_tokens_to_codes(tokens, word2code)


# ### 2.4 Create word by context matrix

# In[294]:

# CAUTION: Think about how big of a matrix will be created...

# how many words to keep in vocabulary (will have one row per vocab word)
max_vocab_words = 50000

# how many words to treat as potential context words (will have one column per context word)
max_context_words = 5000


# In[301]:

t0 = time.time()
X_wiki = generate_word_by_context(codes, 
                                  max_vocab_words=max_vocab_words, 
                                  max_context_words=max_context_words, 
                                  context_size=4,
                                  weight_by_distance=True)
t1 = time.time()
print("elapsed = " + str(t1 - t0) + "s")
# elapsed = 128.49356627464294


# In[304]:

X_wiki[0:4,0:4]


# In[310]:

X_wiki.shape
X_wiki[200,300]


# ### 2.5 Apply SVD

# In[296]:

def reduce(X, n_components, power=0.0):
    U, Sigma, VT = randomized_svd(X, n_components=n_components)
    # note: TruncatedSVD always multiplies U by Sigma, but can tune results by just using U or raising Sigma to a power
    return U * (Sigma**power)


# In[297]:

# apply log to raw counts (which has been shown to improve results)
X_log = np.log10(1 + X_wiki, dtype="float32")


# In[289]:

t0 = time.time()
d = 200
my_vectors = reduce(X_log, n_components=d)
t1 = time.time()
print("elapsed " + str(t1 - t0) + "s")
# elapsed 46.30777978897095s


# Check out first 10 elements of vector for "king":

# In[298]:

my_vectors[word2code["king"]][0:10]


# Save and load back into gensim (so we can use some gensim functionality for exploring our word vectors).

# In[299]:

# save in word2vec format (first line has vocab_size and dimension; other lines have word followed by embedding)
with codecs.open("my_vectors_200.txt", "w", "utf-8") as f:
    f.write(str(max_vocab_words) + " " + str(d) + "\n")
    
    for i in range(max_vocab_words):
        f.write(sorted_words[i] + " " + " ".join([str(x) for x in my_vectors[i,:]]) + "\n")

# load back in
word_vectors = KeyedVectors.load_word2vec_format("my_vectors_200.txt", binary=False)


# ### 2.6 Look at some similar words

# In[250]:

word_vectors.wv.similar_by_word("red")


# In[251]:

word_vectors.wv.similar_by_word("train")


# In[252]:

word_vectors.wv.similar_by_word("last")


# ### 2.7 Try some analogies

# In[254]:

query = word_vectors.wv["paris"] - word_vectors.wv["france"] + word_vectors.wv["germany"]
word_vectors.wv.similar_by_vector(query)


# In[255]:

query = word_vectors.wv["king"] - word_vectors.wv["man"] + word_vectors.wv["woman"]
word_vectors.wv.similar_by_vector(query)


# ### 2.8 Put our vectors to the test!
# 
# The gensim library comes with some methods for qualitatively evaluating how well particular word embeddings do on established benchmark datasets for word similarity and analogy solving. We'll use the gensim's `KeyedVectors.accuracy` method to score our embeddings on a set of analogies from the word2vec researchers.
# 
# Each non-comment line of the file is a tuple of 4 words, e.g.,
# 
#     Athens Greece Baghdad Iraq
# 
# This correponds to the analogy: 
# 
#     "Athens" is to "Greece" as "Baghdad" is to ?
#     
# The accuracy method will try to solve each of the 10000+ analogies in the file, which can take a while. Feel free to derive a shorter set of analogies to speed up the testing if you want.

# In[275]:

# code for unpacking results from: https://gist.github.com/iamaziz/8d8d8c08c7eeda707b9e
def unpack_accuracy(accuracy):
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100   
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))


# In[292]:

t0 = time.time()
results = word_vectors.accuracy("questions-words.txt")
t1 = time.time()
print("elapsed " + str(t1 - t0) + "s")
# elapsed 172.55055022239685s


# In[293]:

unpack_accuracy(results)
# Total sentences: 13965, Correct: 19.10%, Incorrect: 80.90%


# Is the percentage correct good? Note that the form of these analogies are fill in the blank instead of multiple choice (like the style that used to be on the SAT). So random guessing (from our large vocabulary) would do probably do extremely poorly.
# 
# You can compare your results to other systems here: https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)
# 
# Note: gensim's test function will ignore questions where any of the four words aren't in the model's vocabulary. So that can cause the "Total sentences" to vary in the output.

# ### 2.9 Conclusion and extra ideas
# 
# Note: You might notice that the quality isn't as good as when we played around with the pre-trained GloVe embeddings. This isn't too surprising given that we trained on a much smaller number of words (10M vs 6B) and also truncated the X matrix (due to memory limitations) before even applying SVD to find the final embeddings.
# 
# However, experiments have empirically shown that different approaches to learning embeddings result in embeddings that work best for different tasks (i.e., no one approach seems to generate embeddings that are best for every task). In fact, SVD-based embeddings have been shown to perform best on some word similarity tasks.
# 
# In any case, our embeddings definitely show evidence of learning something about word relationships!
# 
# Here are some ideas for other things to do:
# 
# - evaluate pre-trained GloVe embeddings on analogies
# - try higher dimensional GloVe embeddings
# - vary the size of X (before dimensionality reduction) and see the effect on accuracy
# - vary the context size
# - remove stopwords?

# In[271]:

path = "C:/Users/Michael/cogworks/glove/glove.6B.50d.txt.w2v"
t0 = time.time()
glove = KeyedVectors.load_word2vec_format(path, binary=False)
t1 = time.time()
print("elapsed " + str(t1 - t0) + "s")


# In[277]:

glove_results = glove.accuracy("questions-words.txt")
unpack_accuracy(glove_results)


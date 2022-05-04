import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


def tokenize_sentence(sentence):
    return nltk.word_tokenize(sentence)


def stemming_word(word_list):

    ignore_list = ["?", ",", ".", "/", "'", ";", ":", "!"]

    stemm_list = [stemmer.stem(word.lower()) for word in word_list if word not in ignore_list]

    return stemm_list


def bag_of_word(sentence_pattern, all_words):

    sentence_pattern = stemming_word(sentence_pattern)
    matrix = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in sentence_pattern:
            matrix[index] = 1.0

    return matrix









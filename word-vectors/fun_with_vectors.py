import numpy
import scipy
import random
from nltk import bigrams
from math import log2, inf, sqrt
from collections import defaultdict


def get_vector(word, word_vectors):
    word_vec = []
    # Opening and scanning a massive file for word vectors for w1 and w2,
    # for every synonym we want to find may be inefficient, but since my
    # computer only has 4gb of RAM, and the OS, Firefox, and PyCharm
    # take up ~80% of it, I can't maintain a data structure of a > 400gb
    # file in memory without getting a MemoryError
    with open(word_vectors) as f:
        for line in f:
            # replace is for Google's souped-up hyped-up vectors
            s = line.replace('\x1b[?1034h', '').split()
            if s[0].lower() == word:
                word_vec = s[1:]
    return [float(x) for x in word_vec]

def euclidean_distance(w1_vec, w2_vec):
    return scipy.linalg.norm(w1_vec - w2_vec)

def cosine_distance(word1_vec, word2_vec):
    dot_product = word1_vec.dot(word2_vec)
    w1_length = sqrt(sum(x ** 2 for x in word1_vec))
    w2_length = sqrt(sum(x ** 2 for x in word2_vec))
    return dot_product / (w1_length * w2_length)

class DistributionalSemantics:
    def __init__(self, file):
        self.vocab = self._get_vocab(file)
        self.co_occurrence_matrix, self.indices = self._get_co_occurrence_matrix(file)
        self.ppmi_matrix = numpy.zeros((len(self.vocab), len(self.vocab)))

    def _get_vocab(self, file):
        vocab = set()
        with open(file) as f:
            for line in f.readlines():
                vocab.update(set(word for word in line.split()))
        return vocab

    def _get_co_occurrence_matrix(self, file):
        with open(file) as f:
            sents = ' '.join([word.replace('\n', '') for word in f.readlines() if not word == '\n'])
        co_occurrence_matrix = numpy.zeros((len(self.vocab), len(self.vocab)))
        indices = dict()  # for indexing into the co_occurence_matrix by word. Super janky
        windows = [b for b in bigrams(sents.split())]
        i, j = 0, 0
        for word in self.vocab:
            for context in self.vocab:
                # Here we see how many bigrams of the form (word, count) or (count, word) occur
                # Then we apply an add-one smooth and scale the entire matrix by a factor of 10
                co_occurrence_matrix[i, j] = (len([item for item in windows if (item[0] == word and item[1] == context)
                                                   or (item[1] == word and item[0] == context)]) + 1) * 10
                indices[word] = i
                j += 1
            i += 1
            j = 0
        return co_occurrence_matrix, indices

    def compute_ppmi(self):
        corpus_size = numpy.sum(self.co_occurrence_matrix)
        # remember- word_vec and context are index variables
        for word_vec in range(len(self.ppmi_matrix)):
            for context in range(len(self.ppmi_matrix[word_vec])):
                f_ij = self.co_occurrence_matrix[word_vec][context]
                p_ij = f_ij / corpus_size
                p_i = numpy.sum(self.co_occurrence_matrix[word_vec]) / corpus_size
                p_j = numpy.sum(self.co_occurrence_matrix[:, context]) / corpus_size
                self.ppmi_matrix[word_vec][context] = max(log2(p_ij / (p_i * p_j)), 0)
        print()

    def get_vec(self, word):
        return self.co_occurrence_matrix[self.indices[word]], \
               self.ppmi_matrix[self.indices[word]]

    def euclidean_distance(self, word1, word2, reduced=False):
        if reduced:
            reduced_ppmi_matrix = self.ppmi_matrix * self.svd()[2][:, 0:3]
            return scipy.linalg.norm(reduced_ppmi_matrix[self.indices[word1]] -
                                     reduced_ppmi_matrix[self.indices[word2]])
        else:
            return scipy.linalg.norm(self.ppmi_matrix[self.indices[word1]] -
                                     self.ppmi_matrix[self.indices[word2]])

    # Performs an SVD reduction on the PPMI matrix
    def svd(self):
        U, E, Vt = scipy.linalg.svd(self.ppmi_matrix, full_matrices=False)
        U = numpy.matrix(U)
        E = numpy.matrix(numpy.diag(E))
        Vt = numpy.matrix(Vt)
        return U, E, Vt


# This class models a 'synonymy question:' Given a word, pick the value in choices
# that is most similar to the given word.
class Question:
    def __init__(self, word, choices, answer):
        self.choices = choices
        self.answer = answer
        self.word = word  # this may be a bit redundant


class SynonymDetector:
    def __init__(self, synonym_file, word_vectors):
        self.questions = dict()
        self.synsets = defaultdict(list)
        self.word_vectors = word_vectors
        self._create_questions(synonym_file)

    def _create_questions(self, synonym_file):
        self.synsets = self._create_synsets(synonym_file)
        for word in self.synsets:
            answer = random.choice(self.synsets[word])
            choices = self._choose_random(self.synsets, word)
            choices.insert(3, answer)
            self.questions[word] = Question(word, choices, answer)

    def _choose_random(self, synsets, word, n=4):
        l = []
        choices = []
        # todo: figure out how not to do this for every word in synsets
        for val in set(synsets) - {word}:
            choices.append(synsets[val])
        for i in range(n):
            l.append(random.choice(random.choice(choices)))
        return l

    def _create_synsets(self, synonym_file):
        synsets = defaultdict(list)
        with open(synonym_file) as f:
            f.readline()  # scan past column labels
            for line in f.readlines():
                pair = line.replace('to_', '').replace('\n', '').split('\t')
                synsets[pair[0]].append(pair[1])
        return synsets

    # Takes in a word and computes the Euclidean distance and cosine similarity
    # between that word and each multiple-choice option. Returns a Euclidean and
    # cosine guess at the synonym.
    def solve(self, word):
        q = self.questions[word]
        q_vec = numpy.array(get_vector(word, self.word_vectors))
        curr_smallest_euclid = inf
        euclid_choice = ''
        curr_largest_cos = 0
        cos_choice = ''
        for choice in q.choices:
            choice_vec = numpy.array(get_vector(choice, self.word_vectors))
            # For words that aren't in the corpus, we simply don't consider them
            # This is not the best approach, but it is the easiest
            if len(choice_vec) == 0 or len(q_vec) == 0:
                continue
            ed = euclidean_distance(choice_vec, q_vec)
            cd = cosine_distance(choice_vec, q_vec)
            if ed <= curr_smallest_euclid:
                curr_smallest_euclid = ed
                euclid_choice = choice
            if cd >= curr_largest_cos:
                curr_largest_cos = cd
                cos_choice = choice
        return q.answer, euclid_choice, cos_choice

    def answer_all(self, n=None):
        results = []
        i = 0
        for word in self.questions:
            if n is not None and i > n:
                break
            results.append(self.solve(word))
        return results


class Analogy:
    def __init__(self, stem, choices, answer):
        self.stem = stem
        self.choices = choices
        self.answer = answer

class SATAnalogySolver:
    def __init__(self, sat_file, word_vectors):
        # When all you have is a hammer, everything looks like a nail
        self.analogies = self._get_analogies(sat_file)
        self.word_vectors = word_vectors

    def _get_analogies(self, sat_file):
        analogies = []
        alphabet_map = {'a\n':0, 'b\n':1, 'c\n':2, 'd\n':3, 'e\n':4}
        with open(sat_file) as f:
            for line in f:
               if line.startswith('190') or line.startswith('KS') or line.startswith('ML'):
                    lines = []
                    choices = []
                    for i in range(7):
                        lines.append(f.readline())
                    stem = lines[0].split()
                    stem.pop() # disregard lexical category
                    for choice in lines[1:len(lines)-1]:
                        c = choice.split()
                        c.pop()
                        choices.append(c)
                    analogies.append(Analogy(tuple(stem), choices, choices[alphabet_map[lines[len(lines)-1]]]))
        return analogies

    def solve_all(self):
        results_add = []
        results_mult = []
        i = 0
        for analogy in self.analogies:
            if i == 1:
                break
            results_add.append(self._solve(analogy, lambda x, y: x+y))
            results_mult.append(self._solve(analogy, lambda x, y : x*y))
            i += 1
        return results_add, results_mult


    def _solve(self, analogy, combiner):
        left_stem_vec = numpy.array(get_vector(analogy.stem[0], self.word_vectors))
        right_stem_vec = numpy.array(get_vector(analogy.stem[1], self.word_vectors))
        combined_stem = combiner(left_stem_vec, right_stem_vec)
        euclid_choice = []
        cosine_choice = []
        curr_smallest_euclid = inf
        curr_largest_cos = 0
        for choice in analogy.choices:
            left_choice_vec = numpy.array(get_vector(choice[0], self.word_vectors))
            right_choice_vec = numpy.array(get_vector(choice[1], self.word_vectors))
            combined_choice = combiner(left_choice_vec, right_choice_vec)
            ed = euclidean_distance(combined_stem, combined_choice)
            cd = cosine_distance(combined_stem, combined_stem)
            if ed <= curr_smallest_euclid:
                curr_smallest_euclid = ed
                euclid_choice = choice
            if cd >= curr_largest_cos:
                curr_largest_cos = cd
                cosine_choice = choice
        return analogy.stem, analogy.answer, euclid_choice, cosine_choice

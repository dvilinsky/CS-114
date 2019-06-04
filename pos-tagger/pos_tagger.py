# CS114 Spring 2018 Homework 4
# Part-of-speech Tagging with Hidden Markov Models

import os
import re
from collections import defaultdict
from math import log

class POSTagger():

    def __init__(self):
        # you can choose which data structures you want to use
        self.transition = defaultdict(lambda : defaultdict(int))
        self.emission = defaultdict(lambda: defaultdict(int))
        self.START  = '<S>'
        self.STOP = '</S>'
        self.TAG_COUNT = "TAG COUNT"
        self.UNK = 'UNK'

    '''
    Trains a supervised hidden Markov model on a training set.
    Transition probabilities P(s|p)
    Emission probabilities P(w|s)
    '''
    def train(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    sentences = [sent for sent in f.readlines() if not sent == '\n']
                    self._get_matrices(sentences)
        self._get_emission_probabilities()
        self._get_transition_probabilities()

    def _get_transition_probabilities(self):
        for tag in self.transition:
            tag_count = self.transition[tag][self.TAG_COUNT]
            for tag2 in self.transition[tag]:
                self.transition[tag][tag2] = log((self.transition[tag][tag2] + 1)/
                                                 (tag_count + len(self.transition)) )
        for tag in self.transition:
            del self.transition[tag][self.TAG_COUNT]

    def _get_emission_probabilities(self):
        for word in self.emission:
            for tag in self.emission[word]:
                self.emission[word][tag] = log((self.emission[word][tag] + 1)/
                                               (self.transition[tag][self.TAG_COUNT] + len(self.emission)))

    def _get_matrices(self, sentences):
        for sentence in sentences:
            pairs = sentence.split()
            # I am ignoring the ./. token here, because if I don't, P(</S>|./.) will
            # equal 1
            for i in range(len(pairs) - 1):
                curr_tag = pairs[i].split('/')[1]
                curr_word = pairs[i].split('/')[0]
                if i == len(pairs) - 2:
                    next_tag = self.STOP
                else:
                    next_tag = pairs[i+1].split('/')[1]
                if i == 0:
                    self.transition[self.START][curr_tag] += 1
                    self.transition[self.START][self.TAG_COUNT] += 1
                self.transition[curr_tag][next_tag] += 1
                self.transition[curr_tag][self.TAG_COUNT] += 1
                self.emission[curr_word][curr_tag] += 1
        for tag in self.transition:
            #This hopefully says that the UNK word is labeled in the training data
            #with every tag exactly once
            self.emission[self.UNK][tag] = 1

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        v = defaultdict(lambda : defaultdict(float))
        backpointer = []

        # initialization step
        for state in self.transition:
            #We're ignoring these because these either say P(word|state) = 0 or
            #P(state|<s>) = 0, which, in the non-log space viterbi calculation,
            #would cause the result to be 0, and thus ignored
            if self.transition[self.START][state] == 0 or self.emission[sentence[0]][state] == 0:
                pass
            else:
                v[sentence[0]][state] = self.transition[self.START][state] + \
                                       self.emission[sentence[0]][state]
        backpointer.append(self._get_backpointer(v, sentence[0]))

        # recursion step
        prev_word = sentence[0]
        for word in sentence[1:]:
            for state in self.transition:
                #Again we ignore 0 probabilities, because they would make
                #vt-1(j) * p(qj|qi) * p(word|qj) equal 0, and those certainly wouldn't
                #be the max for all i
                if self.emission[word][state] == 0:
                    continue
                x = max(self.transition[state1][state] for state1 in self.transition
                        if not self.transition[state1][state] == 0)
                v[word][state] = v[prev_word][state] + x + self.emission[word][state]
            backpointer.append(self._get_backpointer(v, word))
            prev_word = word
        # So I'm not sure what is supposed to happen here- we've already tagged all the
        #words and come up with a most likely tag sequence
        best_path = backpointer
        return best_path

    #If I understood the point of the backpointer array, I wouldn't need this
    #function. Since I don't, I do.
    def _get_backpointer(self, viterbi, word):
        curr_most_likely_tag = ""
        curr_max = -2000.0 #unlikley log prob gets this low
        for state in viterbi[word]:
            if viterbi[word][state] > curr_max:
                curr_max = viterbi[word][state]
                curr_most_likely_tag = state
        return curr_most_likely_tag

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of POS tags such that:
    results[sentence_id]['correct'] = correct sequence of POS tags
    results[sentence_id]['predicted'] = predicted sequence of POS tags
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    sentences = [sent for sent in f.readlines() if not sent == '\n']
                    sent_tags = dict()
                    for sent in sentences:
                        #Perhaps the ugliest code ever put to disk
                        sent_tags[' '.join([x for x in map(lambda x:x[:re.search(r'/', x).start()], sent.split())])] = ' '.join([l for l in map(lambda x:x[re.search(r'/', x).start() + 1:], sent.split())])
                    for sent in sent_tags:
                        results[sent]['correct'] = sent_tags[sent].split()
                        results[sent]['predicted'] = self.viterbi(sent.split())
        return results

    '''
    Given results, calculates overall accuracy
    '''
    def evaluate(self, results):
        num_right = 0
        for sentence in results:
            if results[sentence]['correct'] == results[sentence]['predicted']:
                num_right += 1
        return num_right/len(results)

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train')
    results = pos.test('brown/dev')
    print(pos.evaluate(results))

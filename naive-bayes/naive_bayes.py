# CS114 Spring 2018 Homework 3
# Naive Bayes Classifier and Evaluation

import os
from collections import defaultdict
from math import log

class NaiveBayes():

    def __init__(self):
        self.prior = defaultdict(float)
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.features = ['hilarious', 'awful', 'funny', 'boring', 'laugh', 'sublime',
                         'interesting','love', 'like', 'film']
        self.word_count = defaultdict(lambda: defaultdict(int))
        self.POS = 'pos'
        self.NEG = 'neg'

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[feature][class] = log(P(feature|class))
    '''
    def train(self, train_set):
        # iterate over training documents
        vocab = set()
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    words = self._get_words(f)
                    vocab.update(words)

                    if root == 'movie_reviews/train\\neg':
                        self.prior[self.NEG] += 1
                        self._get_count(words, self.NEG)
                    elif root == 'movie_reviews/train\\pos':
                        self.prior[self.POS] += 1
                        self._get_count(words, self.POS)
        temp_neg = self.prior[self.NEG]
        self.prior[self.NEG] = log(self.prior[self.NEG]/(self.prior[self.NEG] + self.prior[self.POS]))
        self.prior[self.POS] = log(self.prior[self.POS]/(self.prior[self.POS] + temp_neg))

        for feature in self.features:
            self.likelihood[feature][self.POS] = self._get_probability(feature, self.POS, vocab)
            self.likelihood[feature][self.NEG] = self._get_probability(feature, self.NEG, vocab)

    def _get_probability(self, feature, category, vocab):
        sum = 0
        for word in vocab:
            sum += self.word_count[word][category]
        return log((self.likelihood[feature][category] + 1)/(sum + len(vocab)))

    #Given a list of words and a category type, updates self.word_count and
    #self.likelihood for each word in that category
    def _get_count(self, words, category):
        for word in words:
            if word in self.features:
                self.likelihood[word][category] += 1
            else:
                self.word_count[word][category] += 1

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # calculate log-probabilities for each class
                    # from log-probabilities, get most likely class
                    words = self._get_words(f)
                    pos_prob = self._compute_class_prob(words, self.POS)
                    neg_prob = self._compute_class_prob(words, self.NEG)

                    results[f.name]['predicted'] = self._max(pos_prob, neg_prob)
                    if root == 'movie_reviews/dev\\neg':
                        results[f.name]['correct'] = self.NEG
                    elif root == 'movie_reviews/dev\\pos':
                        results[f.name]['correct'] = self.POS
        return results

    def _max(self, pos_prob, neg_prob):
        #Really don't know what to do if pos_prob = neg_prob, making arbitrary decision
        if pos_prob >= neg_prob:
            return self.POS
        else:
            return self.NEG

    def _get_words(self, f):
        # todo: get rid of ugly double list comp
        words = [line.split() for line in f.readlines()]
        return [word for line in words for word in line]

    def _compute_class_prob(self, words, category):
        sum = 0
        for word in words:
            if word in self.features:
                #I am not taking logs here because the value in the dict
                #is already the log probability
                sum += self.likelihood[word][category]
        #Also not taking logs here because of same reason
        return self.prior[category] + sum

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        recall_pos = self._get_recall(results, self.POS)
        recall_neg = self._get_recall(results, self.NEG)
        precision_pos = self._get_precision(results, self.POS)
        precision_neg = self._get_precision(results, self.NEG)
        accuracy = self._get_accuracy(results)
        f_measure_pos = (2 * precision_pos * recall_pos)/(precision_pos + recall_pos)
        f_measure_neg = (2 * precision_neg * recall_neg)/(precision_neg + recall_neg)
        print("Recall positive: %.4f" % recall_pos)
        print("Recall negative: %.4f" % recall_neg)
        print("Precision positive: %.4f" %  precision_pos)
        print('Precision negative: %.4f' % precision_neg)
        print('Accuracy:', accuracy)
        print('F-measure positive: %.4f' % f_measure_pos)
        print('F-meausre negative: %.4f' % f_measure_neg)

    # results[filename]['correct'] = correct class
    # results[filename]['predicted'] = predicted class
    def _get_recall(self, results, category):
        num_right = 0
        total = 0 #A bit redundant to recalculate this
        for result in results.values():
            if result['correct'] == category:
                if result['correct'] == result['predicted']:
                    num_right += 1
                total += 1
        return num_right/total

    def _get_precision(self, results, category):
        num_right = 0
        total = 0
        for result in results.values():
            if result['predicted'] == category:
                total += 1
                if result['correct'] == result['predicted']:
                    num_right += 1
        return num_right/total

    def _get_accuracy(self, results):
        num_right = 0
        for result in results.values():
            if result['correct'] == result['predicted']:
                num_right += 1
        return num_right/len(results)

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)

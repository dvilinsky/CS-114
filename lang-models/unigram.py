from collections import defaultdict
from languageModel import LanguageModel
import random
import bisect

'''
Tuan Do, Kenneth Lai
'''
class Unigram(LanguageModel):

    def __init__(self):
        # P(word) = self.probCounter[word]
        self.probCounter = defaultdict(float)
        self.rand = random.Random()
    
    def train(self, trainingSentences):
        # accu contains accumulated counts of words
        # e.g. if words are ['the', 'dog', 'barked'], in that order,
        # and c('the') = 4, c('dog') = 3, and c('barked') = 3,
        # then self.accu = [4, 7, 10]
        self.accu = []
        self.total = 0
        for sentence in trainingSentences:
            for word in sentence:
                # count words
                self.probCounter[word] += 1
                self.total += 1
            self.probCounter[LanguageModel.STOP] += 1
            self.total += 1

        self.probCounter[LanguageModel.UNK] += 1
        self.total += 1
            
        for word in self.probCounter.keys():
            self.accu.append(self.probCounter[word]
                             if len(self.accu) == 0
                             else self.accu[-1] + self.probCounter[word])
            # divide counts by total to get probabilities
            self.probCounter[word] /= self.total


    def getWordProbability(self, sentence, index):
        if index == len(sentence):
            return self.probCounter[LanguageModel.STOP]
        else:
            word = sentence[index]
            return (self.probCounter[word]
                    if word in self.probCounter
                    else self.probCounter[LanguageModel.UNK])

    def getVocabulary(self, context):
        return self.probCounter.keys()

    def generateWord(self):
        # generate random integer
        i = self.rand.randint(0, self.total - 1)
        # use it to pick a word index from self.accu
        index = bisect.bisect_left( self.accu, i )
        # return corresponding word
        return list(self.getVocabulary([]))[index]

    def generateSentence(self):
        result = []
        # limit sentence length to 1000
        for i in range(1000):
            word = self.generateWord()
            result.append(word)
            if word == LanguageModel.STOP:
                break
        return result

    def get_probability(self):
        return self.probCounter

if __name__ == '__main__':
    u = Unigram()
    trainingSentences = 'data/train-data.txt'
    with open(trainingSentences, 'r') as f:
        sentences = [line.split() for line in f.readlines()]
    u.train(sentences)

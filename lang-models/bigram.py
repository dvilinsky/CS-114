from languageModel import LanguageModel
from collections import defaultdict
import bisect
import random

class Bigram(LanguageModel):
    def __init__(self):
        self.bigram_count = defaultdict(int)
        self.word_count = defaultdict(int)  # for use in MLE
        self.total_bigrams = 0

    def train(self, trainingSentences):
        bigrams = self._get_bigrams(trainingSentences)
        for bigram in bigrams:
            self.total_bigrams += 1
            self.bigram_count[bigram] += 1

        self.bigram_count[LanguageModel.UNK] += 1

    def _get_bigrams(self, trainingSentences):
        self.total_words = 0
        bigrams = []
        for sent in trainingSentences:
            self.total_words += len(sent)
            for i in range(len(sent)):
                self.word_count[sent[i]] += 1
                if i == 0:
                    bigrams.append((LanguageModel.START, sent[i]))
                    self.word_count[LanguageModel.START] += 1
                if i <= len(sent) - 2:
                    bigrams.append(tuple(sent[i:i + 2]))
                if i == len(sent) - 1:
                    bigrams.append((sent[i], LanguageModel.STOP))
                    self.word_count[LanguageModel.STOP] += 1
        self.word_count[LanguageModel.UNK] = 3  # same three cases as in self.train
        return bigrams

    #Basic unsmoothed MLE. May return zero
    def getWordProbability(self, sentence, index):
        bigram = self.get_bigram(sentence, index)
        bigram_count = self.bigram_count[bigram]
        count_previous = self.word_count[bigram[0]]
        return count_previous if count_previous == 0 else bigram_count/count_previous

    def get_bigram(self, sentence, index):
        if index == 0:
            previous_word = LanguageModel.START
            next_word = sentence[index]
        elif index == len(sentence):
            previous_word = sentence[index - 1]
            next_word = LanguageModel.STOP
        else:
            previous_word = sentence[index - 1]
            next_word = sentence[index]
        return (previous_word, next_word)

    def generateSentence(self):
        sentence = []
        previous = LanguageModel.START
        # the range is small so as not to throw a memory error-
        # but that might be machine-dependent. If you have a better machine,
        # you may increase the range here
        for i in range(20):
            word = self.generate_word(previous)
            sentence.append(word)
            previous = word
            if word == LanguageModel.STOP:
                break
        return sentence

    def generate_word(self, previous):
        accumulator, num_with_previous = self.create_accumulator(previous)
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_previous))
        return list(self.word_count.keys())[index]

    def create_accumulator(self, previous, smooth_value=0, bucket_start=0):
        current_bucket = bucket_start
        bucket_list = []
        num_with_previous = 0
        for word2 in self.word_count.keys():
            if self.bigram_count[(previous, word2)] == 0:
                current_bucket += smooth_value
            else:
                current_bucket += self.bigram_count[(previous, word2)] + smooth_value
                num_with_previous += self.bigram_count[(previous, word2)]
            bucket_list.append(current_bucket)
        return bucket_list, num_with_previous

class BigramAddOneSmooth(Bigram):
    def __init__(self):
        super(BigramAddOneSmooth, self).__init__()

    def train(self, trainingSentences):
       super().train(trainingSentences)


    def getVocabulary(self, context):
        return self.word_count.keys()

    # Maximum likelihood estimate with add-1 smoothing
    # P(word|previous) = count(previous, word)+1/count(previous) + |V|
    def getWordProbability(self, sentence, index):
        bigram = super().get_bigram(sentence, index)
        if bigram in self.bigram_count:
            bigram_count = self.bigram_count[bigram]
        else:
            bigram_count = 0
        if bigram[0] in self.word_count:
            count_previous = self.word_count[bigram[0]]
        else:
            count_previous = 0
        return (bigram_count + 1)/(count_previous + len(self.word_count))


    def generateSentence(self):
        sentence = []
        previous = LanguageModel.START
        # the range is small so as not to throw a memory error-
        # but that might be machine-dependent. If you have a better machine,
        # you may increase the range here
        for i in range(20):
            word = self.generate_word(previous)
            sentence.append(word)
            previous = word
            if word == LanguageModel.STOP:
                break
        return sentence


    def generate_word(self, previous):
        accumulator, num_with_previous = super().create_accumulator(previous, smooth_value=1, bucket_start=-1)
        num_with_previous +=  len(self.word_count.keys()) - 1
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_previous))
        return list(self.word_count.keys())[index]

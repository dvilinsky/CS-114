from languageModel import LanguageModel
from collections import defaultdict
import random
import bisect

'''
Implements a basic trigram language model with no smoothing. Not directly useful,
used for interpolation mostly
'''
class Trigram(LanguageModel):
    def __init__(self):
        self.trigram_count = defaultdict(int)
        self.word_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        self.total_trigrams = 0

    def train(self, trainingSentences):
        trigrams = self._get_trigrams(trainingSentences)
        bigrams = self._get_bigrams(trainingSentences)
        for trigram in trigrams:
            self.trigram_count[trigram] += 1
            self.total_trigrams += 1
        for bigram in bigrams:
            self.bigram_count[bigram] += 1

        self.trigram_count[LanguageModel.UNK] += 1
        self.bigram_count[LanguageModel.UNK] += 1

    def _get_trigrams(self, trainingSentences):
        self.total_words = 0
        trigrams = []
        for sent in trainingSentences:
            self.total_words += len(sent)
            for i in range(len(sent)):
                self.word_count[sent[i]] += 1
                if len(sent) == 1:
                    # To keep consistent with our strategy in _get_trigram
                    trigrams.append((LanguageModel.START, LanguageModel.START, sent[0]))
                    trigrams.append((sent[0], LanguageModel.STOP, LanguageModel.STOP))
                    self.word_count[LanguageModel.START] += 2
                    self.word_count[LanguageModel.STOP] += 2
                elif len(sent) == 2:
                    trigrams.append((LanguageModel.START, sent[0], sent[1]))
                    trigrams.append((sent[0], sent[1], LanguageModel.STOP))
                    self.word_count[LanguageModel.START] += 1
                    self.word_count[LanguageModel.STOP] += 1
                elif i == 0:
                    trigrams.append((LanguageModel.START, sent[i], sent[i+1]))
                    self.word_count[LanguageModel.START] += 1
                elif i == len(sent) - 1:
                    trigrams.append((sent[i-1], sent[i], LanguageModel.STOP))
                    self.word_count[LanguageModel.STOP] += 1
                else:
                    trigrams.append(tuple(sent[i:i+3]))
        self.word_count[LanguageModel.UNK] = 7
        return trigrams

    def _get_bigrams(self, trainingSentences):
        bigrams = []
        for sent in trainingSentences:
            for i in range(len(sent)):
                if i == 0:
                    bigrams.append((LanguageModel.START, sent[i]))
                if i <= len(sent) - 2:
                    bigrams.append(tuple(sent[i:i + 2]))
                if i == len(sent) - 1:
                    bigrams.append((sent[i], LanguageModel.STOP))
        return bigrams

    #Unsmoothed MLE. May return 0
    def getWordProbability(self, sentence, index):
        trigram = self.get_trigram(sentence, index)
        bigram = (trigram[0], trigram[1])
        trigram_count = self.trigram_count[trigram]
        bigram_count = self.bigram_count[bigram]
        return bigram_count if bigram_count == 0 else trigram_count/bigram_count


    def get_trigram(self, sentence, index):
        #Wikipedia is using two start symbols for the context of the first word.
        #That's good enough for me
        if index == 0:
            prev_prev_word = LanguageModel.START
            prev_word = LanguageModel.START
            next_word = sentence[index]
        elif index == 1:
            prev_prev_word = LanguageModel.START
            #In an ideal world, a client of this class wouldn't ask for p(word) at
            #a non-existent position in a sentence. But we don't live in that world
            if len(sentence) == 1:
                prev_word = LanguageModel.START
                next_word = sentence[index-1]
            else:
                prev_word = sentence[index-1]
                next_word = sentence[index]
        elif index == len(sentence):
            prev_prev_word = sentence[index-2]
            prev_word = sentence[index-1]
            next_word = LanguageModel.STOP
        else:
            prev_prev_word = sentence[index-2]
            prev_word = sentence[index-1]
            next_word = sentence[index]
        return (prev_prev_word, prev_word, next_word)

    def generateSentence(self):
        sentence = []
        prev_previous = LanguageModel.START
        # Maybe not the best choice, but we need to choose a second context
        previous = random.choice(list(self.word_count.keys()))
        for i in range(20):
            word = self.generateWord(prev_previous, previous)
            if word == LanguageModel.STOP:
                break
            prev_previous = previous
            previous = word
            sentence.append(word)
        return sentence

    def generateWord(self, prev_previous, previous):
        accumulator, num_with_previous = self.create_accumulator(prev_previous, previous)
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_previous))
        return list(self.word_count.keys())[index]

    def create_accumulator(self, prev_previous, previous, smooth_value=0, start=0):
        buckets = []
        current_bucket = start
        occurrences = 0
        for word in self.word_count.keys():
            if self.trigram_count[(prev_previous, previous, word)] == 0:
                current_bucket += smooth_value
            else:
                current_bucket += self.trigram_count[(prev_previous, previous, word)]
                occurrences += self.trigram_count[(prev_previous, previous, word)]
            buckets.append(current_bucket)
        return buckets, occurrences

    def getVocabulary(self, context):
        return self.word_count.keys()

class TrigramAddOneSmooth(Trigram):
    def __init__(self):
        super().__init__()

    def train(self, trainingSentences):
       super().train(trainingSentences)

    def getWordProbability(self, sentence, index):
        trigram = super().get_trigram(sentence, index)
        bigram = (trigram[0], trigram[1])
        if trigram in self.trigram_count:
            trigram_count = self.trigram_count[trigram]
        else:
            trigram_count = 0
        if bigram in self.bigram_count:
            bigram_count = self.bigram_count[bigram]
        else:
            bigram_count = 0
        return float((trigram_count + 1))/(bigram_count + len(self.word_count))

    def getVocabulary(self, context):
        return self.word_count.keys()

    def generateSentence(self):
        sentence = []
        prev_previous = LanguageModel.START
        #Maybe not the best choice
        previous = random.choice(list(self.word_count.keys()))
        for i in range(20):
            word = self.generateWord(prev_previous, previous)
            if word == LanguageModel.STOP:
                break
            prev_previous = previous
            previous = word
            sentence.append(word)
        return sentence

    def generateWord(self, prev_previous, previous):
        accumulator, num_with_context = super().create_accumulator(
            prev_previous, previous, smooth_value=1, start=-1)
        num_with_context += len(self.word_count.keys()) - 1
        index = bisect.bisect_left(accumulator, random.randint(1, num_with_context))
        return list(self.word_count.keys())[index]
if __name__ == '__main__':
    t = TrigramAddOneSmooth()
    train_path = 'data/train-data.txt'
    with open(train_path, 'r') as fh:
        sentences = [line.split() for line in fh.readlines()]
    t.train(sentences)
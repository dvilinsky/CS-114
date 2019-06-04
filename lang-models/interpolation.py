from languageModel import LanguageModel
from unigram import Unigram
from bigram import BigramAddOneSmooth
from trigram import TrigramAddOneSmooth
import random

class Interpolation(LanguageModel):
    def __init__(self):
        self.unigram_model = Unigram()
        self.bigram_model = BigramAddOneSmooth()
        self.trigram_model = TrigramAddOneSmooth()
        self.unigram_lambda = .25
        self.bigram_lambda = .25
        self.trigram_lambda = .5

    def train(self, trainingSentences):
        self.unigram_model.train(trainingSentences)
        self.bigram_model.train(trainingSentences)
        self.trigram_model.train(trainingSentences)

    #Arbitrary lambdas.
    def getWordProbability(self, sentence, index):
        return (self.trigram_lambda * self.trigram_model.getWordProbability(sentence, index)) \
               + (self.bigram_lambda * self.bigram_model.getWordProbability(sentence, index)) \
               + (self.unigram_lambda * self.unigram_model.getWordProbability(sentence, index))

    #Doesn't matter which model we use here- vocabulary is the same
    def getVocabulary(self, context):
        return self.trigram_model.getVocabulary(context)

    #What does generating a sentence in an interpolation model look like?
    #I don't know, so what I've done is generate a word using trigram, bigram, and
    #unigram model some of the time, using the same values in getWordProbability
    def generateSentence(self):
        sentence = []
        prev_previous = LanguageModel.START
        previous = random.choice(list(self.trigram_model.word_count.keys()))
        for i in range(20):
            model_choice = random.random()
            if model_choice <= self.trigram_lambda:
                word = self.trigram_model.generateWord(prev_previous, previous)
            elif model_choice > self.trigram_lambda and model_choice <= self.trigram_lambda + self.bigram_lambda:
                word = self.bigram_model.generate_word(previous)
            else:
                word = self.unigram_model.generateWord()
            sentence.append(word)
            prev_previous = previous
            previous = word
            if word == LanguageModel.STOP:
                break
        return sentence
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:59:44 2020

@author: wenjie zhang
""" 
from __future__ import division
import sys, getopt, re, pandas as pd, seaborn, io, nltk, numpy as np, string
from nltk import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from collections import Counter 
from collections import defaultdict
from numpy import mean
#download
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


#read file
class FileLoader:
    def __init__(self, File):
        #read file
        self.data = pd.read_csv(File, sep='\t', encoding='utf8')   
    def preprocess(self):
        # tried lots of preprocess methods but somehow these preprocess
        # decrease accuracy for both 3 and 5 value sentiment. so only use 3 preprocess
        # e.g ignore "," in text as below
        # self.data['Phrase'] = self.data['Phrase'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) )
        
        #stemming preprocess
        snowball = SnowballStemmer(language = 'english')
        self.data['Phrase'] = self.data['Phrase'].apply(lambda x: ' '.join([snowball.stem(word) for word in x.split()]))
        # stoplist preprocess
        stop_words = set(stopwords.words('english'))         
        self.data['Phrase'] = self.data['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        #lowecasing preprocess
        self.data['Phrase'] = self.data['Phrase'].str.lower()
    
    #transform 5 value sentiment into 3 value
    def trans_senti(self):
        def fivetothree(x):
            if x <= 1:
                return 0
            elif x == 2:
                return 1
            else:
                return 2
        self.data['Sentiment'] = self.data['Sentiment'].apply(fivetothree)
        return self.data
    # use for 5-value sentiment and get data  
    def getdata(self):
        return self.data

# extract features
class FeatureSelection:
    def __init__(self, data):
        self.data = data
   
    # 1 methods for extracting features
    # 1.1 take all word as feature
    def bag_of_words(self):
        # convert phrase into strings in order to adapt bigrams
        def Convert(strings): 
            li = list(strings.split(" ")) 
            return li 
        self.data['Phrase'] = self.data['Phrase'].apply(Convert)
        
        return self.data


    # 1.2 take bigrams as features
    def bigram(self, score_fn=BigramAssocMeasures.chi_sq, n=1000):
       def phrase_to_feature(x):
           # transform text into bigrams form
           bigram_finder = BigramCollocationFinder.from_words(x.split())  
           # take first 1000 bigrams, score calculated by BigramAssocMeasures.chi_sq
           bigrams = bigram_finder.nbest(score_fn, n) 
           return bigrams
       self.data['Phrase'] = self.data['Phrase'].apply(phrase_to_feature)
        
       return self.data
   # 1.3 take bigrams and all words together as feature 
    def bigram_words(self, score_fn=BigramAssocMeasures.chi_sq, n=1000):
        def phrase_to_feature1(x):
            # get all words
            tuple_words = []
            for word in x.split():
                temp = (word,)
                tuple_words.append(temp)
            # take first 1000 bigrams 
            bigram_finder = BigramCollocationFinder.from_words(x.split())
            bigrams = bigram_finder.nbest(score_fn, n) 
            return (tuple_words + bigrams)
        self.data['Phrase'] = self.data['Phrase'].apply(phrase_to_feature1)
        
        return self.data  

# Bayes clssifier
class BayesClassifier: 
    def __init__(self, num):
        # create a counter for each sentiment value
        self.num_value_senti = [Counter() for i in range(num)]
        # use set to collect all features
        # since set dont allow duplicate
        self.num_unique_feature = set()
        # count frequency of each class
        self.prior_frequency = Counter()
        # how many unique sentiment value
        self.sentiment_num = num
                
    def train_model(self, words, sentiment):
        self.prior_frequency[sentiment] += 1
        # count frequency of each word in specific class
        for word in words:
            self.num_value_senti[sentiment][word] += 1
            self.num_unique_feature.add(word)
            
    # classify sentiment      
    def classify(self, words): 
        # initialise possibility and sentiment
        max_possibility = 0
        max_senti_value = 0
        # for given phrase, compute possibility for each class 
        for sent in range(self.sentiment_num):
            prior = self.prior_frequency[sent] / train_data.getdata().shape[0]
            for word in words:
                prior *=  (self.num_value_senti[sent][word] + 1) / (sum(self.num_value_senti[sent].values()) + len(self.num_unique_feature))
            # if current possibility is biggest then set it to max possibility
            if prior > max_possibility:
                # set sentiment to this class
                max_senti_value = sent
                max_possibility = prior
        return max_senti_value         

#==============================================================================
# MAIN

# 5 sentiment value: 0, 1, 2, 3, 4
sentiment_value5 = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]
# 3 sentiment value: 0, 1, 2
sentiment_value3 = ["negative", "neutral", "posotive"]

# initialise 3 and 5 value classifier
sentiment3 = BayesClassifier(3)
sentiment5 = BayesClassifier(5)

# 5 value sentiment
# load data and preprocess
train_data = FileLoader('train.tsv') 
train_data.preprocess()
# transform phrase into features, which is used to train model   
featured_data = FeatureSelection(train_data.getdata())
# all words is used here
all_word = featured_data.bag_of_words()
all_word.apply(lambda row: sentiment5.train_model(row['Phrase'], row['Sentiment']), axis=1)
#load test data
test_data = FileLoader('dev.tsv') 
test_data.preprocess()  
# transform phrase into features, which is used to train model  
featured_test_data = FeatureSelection(test_data.getdata())
# all words is used here
all_word_test = featured_test_data.bag_of_words() 
# classify and create new colunms to store results, one for T/F one for sentiment
all_word_test['myClassification5'] = all_word_test.apply(lambda row : sentiment5.classify(row['Phrase']), axis=1)
all_word_test['Match5'] = all_word_test.apply(lambda row: row['Sentiment'] == row['myClassification5'], axis=1) # check if the classification match the real result
print("For 5 sentiments: \n\t", all_word_test[['Sentiment', 'Match5']].groupby('Match5').agg(['count'])) # do a brief summary

# 3 value sentiment, same again
train_data1 = FileLoader('train.tsv')  
train_data1.preprocess()
featured_data1 = FeatureSelection(train_data1.trans_senti())

all_word1 = featured_data1.bag_of_words()
all_word1.apply(lambda row: sentiment3.train_model(row['Phrase'], row['Sentiment']), axis=1)
test_data1 = FileLoader('dev.tsv') 
test_data1.preprocess()  
featured_test_data1 = FeatureSelection(test_data1.trans_senti())
all_word_test1 = featured_test_data1.bag_of_words() 

all_word_test1['myClassification3'] = all_word_test1.apply(lambda row : sentiment3.classify(row['Phrase']), axis=1)
all_word_test1['Match3'] = all_word_test1.apply(lambda row: row['Sentiment'] == row['myClassification3'], axis=1) # check if the classification match the real result
print("For 3 sentiments: \n\t", all_word_test1[['Sentiment', 'Match3']].groupby('Match3').agg(['count'])) # do a brief summary


# write result to file
all_word_test[['SentenceId', 'myClassification5']].to_csv("dev_predictions_5classes_Wenjie_ZHANG.tsv", sep='\t', index=False)
all_word_test1[['SentenceId', 'myClassification3']].to_csv("dev_predictions_3classes_Wenjie_ZHANG.tsv", sep='\t', index=False)


test = FileLoader('test.tsv')  
test.preprocess()
featured_test = FeatureSelection(test.getdata())
all_word_test_data = featured_test.bag_of_words()
all_word_test_data['myClassification3'] = all_word_test_data.apply(lambda row : sentiment3.classify(row['Phrase']), axis=1)
all_word_test_data['myClassification5'] = all_word_test_data.apply(lambda row : sentiment5.classify(row['Phrase']), axis=1)

all_word_test_data[['SentenceId', 'myClassification5']].to_csv("test_predictions_5classes_Wenjie_ZHANG.tsv", sep='\t', index=False)
all_word_test_data[['SentenceId', 'myClassification3']].to_csv("test_predictions_3classes_Wenjie_ZHANG.tsv", sep='\t', index=False)

     
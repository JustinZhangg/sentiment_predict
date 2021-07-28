# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:59:44 2020

@author: wenjie zhang
""" 
from __future__ import division # make sure we do precise division
import sys, getopt, re, pandas as pd, seaborn, io, nltk, numpy as np, string
from nltk import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from collections import Counter # Counter() is a dict for counting
from collections import defaultdict
from numpy import mean
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
#print(stopwords.words('english'))
#read file 
  
def preprocess(data):
        # ignore "," in text
        data['Phrase'] = data['Phrase'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) )
        #stemming preprocess
        snowball = SnowballStemmer(language = 'english')
        data['Phrase'] = data['Phrase'].apply(lambda x: ' '.join([snowball.stem(word) for word in x.split()]))
        # stoplist preprocess
        stop_words = set(stopwords.words('english'))         
        data['Phrase'] = data['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        #lowecasing preprocess
        data['Phrase'] = data['Phrase'].str.lower()
        return data
#data = pd.read_csv('train.tsv', sep='\t', encoding='utf8')
     
def trans_senti(data):
            if data <= 1:
                return 0
            elif data == 2:
                return 1
            else:
                return 2
        
            return data

    # 1 提取特征方法
   # 1.1 把所有词作为特征
def bag_of_words(data):
        # convert phrase into strings in order to adapt bigrams
        
        def Convert(strings): 
            li = list(strings.split(" ")) 
            return li 
        data['Phrase'] = data['Phrase'].apply(Convert)
        
        return data


    # 1.2 把双词搭配（bigrams）作为特征
def bigram(data, score_fn=BigramAssocMeasures.chi_sq, n=1000):
       def phrase_to_feature(x):
           # transform text into bigrams form
           bigram_finder = BigramCollocationFinder.from_words(x.split())  
           # take first 1000 bigrams 
           bigrams = bigram_finder.nbest(score_fn, n) 
           return bigrams
       data['Phrase'] = data['Phrase'].apply(phrase_to_feature)
        
       return data

def bigram_words(data, score_fn=BigramAssocMeasures.chi_sq, n=1000):
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
        data['Phrase'] = data['Phrase'].apply(phrase_to_feature1)
        
        return data  

class BayesClassifier: # A counter counts for all sentiment and the whole situation.
    def __init__(self, num): # number of counter
        self.num_value_senti = [Counter() for i in range(num)]
        self.num_unique_feature = set()
        self.prior_frequency = Counter()
        self.sentiment_num = num
        #self.lenth = 0
        
    def train_model(self, words, sentiment):
        self.prior_frequency[sentiment] += 1
        #self.lenth += 1
        for word in words:
            self.num_value_senti[sentiment][word] += 1
            self.num_unique_feature.add(word)
            
    def classify(self, ws): # given a list of words, classify the sentiment
        max_prior = 0
        max_senti_value = 0
        for sent in range(self.sentiment_num):
            prior = self.prior_frequency[sent] / train_data.shape[0]
            for word in ws:
                prior *=  (self.num_value_senti[sent][word] + 1) / (sum(self.num_value_senti[sent].values()) + len(self.num_unique_feature))
            if prior > max_prior:
                max_senti_value = sent
                max_prior = prior
        return max_senti_value         

#==============================================================================
# MAIN
# load data
sentiment5 = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]


# 3 sentiment value: 0, 1, 2
sentiment3 = ["negative", "neutral", "posotive"]
counter3 = BayesClassifier(3)
counter5 = BayesClassifier(5)

train_data = pd.read_csv("train.tsv", sep='\t')
preprocessed_data = preprocess(train_data)

featured_data = bag_of_words(preprocessed_data)
# now we get a series of preprocessed word lists
featured_data.apply(lambda row: counter5.train_model(row['Phrase'], row['Sentiment']), axis=1)
featured_data.apply(lambda row: counter3.train_model(row['Phrase'], trans_senti(row['Sentiment'])), axis=1)

def classify(counter5, counter3, data):
    preprocessed_data = preprocess(data)
    featured_data = bag_of_words(preprocessed_data)
    return featured_data.apply(lambda row : counter5.classify(row['Phrase']), axis=1), featured_data.apply(lambda row : counter3.classify(row['Phrase']), axis=1)

def dev_check(counter5, counter3, dev_data):
    dev_data['myClassification5'], dev_data['myClassification3'] = classify(counter5, counter3, dev_data)
    dev_data['Match5'] = dev_data.apply(lambda row: row['Sentiment'] == row['myClassification5'], axis=1) # check if the classification match the real result
    dev_data['Match3'] = dev_data.apply(lambda row: row['Sentiment'] == row['myClassification3'], axis=1) # check if the classification match the real result
    print("For 5 sentiments: \n\t", dev_data[['Sentiment', 'Match5']].groupby('Match5').agg(['count'])) # do a brief summary
    print("For 3 sentiments: \n\t", dev_data[['Sentiment', 'Match3']].groupby('Match3').agg(['count'])) # do a brief summary

dev_data = pd.read_csv("dev.tsv", sep='\t')
dev_check(counter5, counter3, dev_data)
























     
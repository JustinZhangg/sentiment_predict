# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:59:44 2020

@author: wenjie zhang
""" 
import sys, getopt, re, pandas as pd, seaborn, io, nltk, numpy as np, string
from nltk import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
#print(stopwords.words('english'))
#read file 

class FileLoader:
    def __init__(self, File):
        #read file
        self.data = pd.read_csv(File, sep='\t', encoding='utf8')
    
    def preprocess(self):
        # ignore "," in text
        self.data['Phrase'] = self.data['Phrase'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) )
        #stemming preprocess
        snowball = SnowballStemmer(language = 'english')
        self.data['Phrase'] = self.data['Phrase'].apply(lambda x: ' '.join([snowball.stem(word) for word in x.split()]))
        # stoplist preprocess
        stop_words = set(stopwords.words('english'))         
        self.data['Phrase'] = self.data['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        #lowecasing preprocess
        self.data['Phrase'] = self.data['Phrase'].str.lower()
        return self.data
    def three_senti(self):
        def fivetothree(x):
            if x <= 1:
                return 0
            elif x == 2:
                return 1
            else:
                return 2
        self.data['3-Sentiment'] = self.data['Sentiment'].apply(fivetothree)
        return self.data
    # use for 5-value trainning and getdata in test 
    def getdata(self):
        return self.data
    #def get_phrase(self):
   
class FeatureSelection:
    def __init__(self, data):
        self.data = data
   
    # 1 提取特征方法
   # 1.1 把所有词作为特征
    def bag_of_words(self):
        # convert phrase into strings in order to adapt bigrams
        def Convert(string): 
            li = list(string.split(" ")) 
            return li 
        self.data['Phrase'] = self.data['Phrase'].apply(Convert)
        return self.data


    # 1.2 把双词搭配（bigrams）作为特征
    def bigram(self, score_fn=BigramAssocMeasures.chi_sq, n=10):
       def phrase_to_feature(x):
           # transform text into bigrams form
           bigram_finder = BigramCollocationFinder.from_words(x.split())  
           # take first 1000 bigrams 
           bigrams = bigram_finder.nbest(score_fn, n) 
           return bigrams
       self.data['Phrase'] = self.data['Phrase'].apply(phrase_to_feature)
        
       return self.data

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

class Train:
    def __init__(self,data):
        self.data = data
        self.total_num_feature = 0
        self.num_very_pos_feature = 0
        self.num_pos_feature = 0
        self.num_very_neg_feature = 0
        self.num_neg_feature = 0
        self.num_neu_feature = 0
        self.very_pos_features = dict()
        self.pos_features = dict()
        self.very_neg_features =  dict()
        self.neg_features =  dict()
        self.neu_features = dict()
        self.very_pos_prior = 0
        self.pos_prior = 0
        self.neg_prior = 0
        self.very_nag_prior = 0
        self.neu_prior = 0
     
    def priorProbability(self):
        # filter function 
        is_value0 = self.data['Sentiment'] == 0
        is_value1 = self.data['Sentiment'] == 1
        is_value2 = self.data['Sentiment'] == 2
        is_value3 = self.data['Sentiment'] == 3
        is_value4 = self.data['Sentiment'] == 4
        # calculate prior probabilty
        self.very_pos_prior = self.data[is_value4].shape[0] / self.data.shape[0]
        self.pos_prior = self.data[is_value3].shape[0]  / self.data.shape[0]
        self.neg_prior = self.data[is_value1].shape[0]  / self.data.shape[0]
        self.very_nag_prior = self.data[is_value0].shape[0] / self.data.shape[0]
        self.neu_prior = self.data[is_value2].shape[0]  / self.data.shape[0]
       
    def likelyhood(self):
        # filter function 
        is_value0 = self.data['Sentiment'] == 0
        is_value1 = self.data['Sentiment'] == 1
        is_value2 = self.data['Sentiment'] == 2
        is_value3 = self.data['Sentiment'] == 3
        is_value4 = self.data['Sentiment'] == 4
        #very_pos_features = {element: 0 for element in self.features}
        #use set to get all different features
        total_features = set()        
        for phrase in self.data['Phrase']:
            for feature in phrase:
                total_features.add(feature)
        self.total_num_feature = len(total_features)       
        # add all features and initilise to 0
        self.very_pos_features = dict.fromkeys(total_features,0)
        self.pos_features = dict.fromkeys(total_features,0)
        self.very_neg_features =  dict.fromkeys(total_features,0)
        self.neg_features =  dict.fromkeys(total_features,0)
        self.neu_features = dict.fromkeys(total_features,0)
        
        #calculate frequency of each feature in their class
        for phrase in self.data[is_value4]['Phrase']:
            for feature in phrase:
                if feature in self.very_pos_features:
                    self.very_pos_features[feature] += 1
                    self.num_very_pos_feature += 1       
                else:
                    self.very_pos_features[feature] = 1
                    
        for phrase in self.data[is_value3]['Phrase']:
            for feature in phrase:
                if feature in self.pos_features:
                    self.pos_features[feature] += 1
                    self.num_pos_feature += 1    
                else:
                    self.pos_features[feature] = 1      
        for phrase in self.data[is_value2]['Phrase']:
            for feature in phrase:
                if feature in self.neu_features:
                    self.neu_features[feature] += 1
                    self.num_neu_feature += 1    
                else:
                    self.neu_features[feature] = 1             
                    
        for phrase in self.data[is_value1]['Phrase']:
            for feature in phrase:
                if feature in self.neg_features:
                    self.neg_features[feature] += 1
                    self.num_neg_feature += 1    
                else:
                    self.neg_features[feature] = 1  
        for phrase in self.data[is_value0]['Phrase']:
            for feature in phrase:
                if feature in self.very_neg_features:
                    self.very_neg_features[feature] += 1
                    self.num_very_neg_feature += 1    
                else:
                    self.very_neg_features[feature] = 1  
               
        # transfer frequency into likelyhood
        for feature in self.very_pos_features:
             (self.very_pos_features[feature] + 1) / (self.num_very_pos_feature + self.total_num_feature)
        for feature in self.pos_features:
             (self.pos_features[feature] + 1) / (self.num_pos_feature + self.total_num_feature)
        for feature in self.very_neg_features:
             (self.very_neg_features[feature] + 1) / (self.num_very_neg_feature + self.total_num_feature) 
        for feature in self.neg_features:
             (self.neg_features[feature] + 1) / (self.num_neg_feature + self.total_num_feature) 
        for feature in self.neu_features:
             (self.neu_features[feature] + 1) / (self.num_neu_feature + self.total_num_feature)
                     
    def get_model(self):
          return (self.very_pos_features, self.pos_features, 
                  self.very_neg_features, self.neg_features,
                  self.neu_features,self.very_pos_prior, self.pos_prior,
                  self.neg_prior, self.very_nag_prior, self.neu_prior, self.total_num_feature,
                  self.num_very_pos_feature ,
                  self.num_pos_feature ,
                  self.num_very_neg_feature,
                  self.num_neg_feature ,
                  self.num_neu_feature )

class BayesClassifier:
    def __init__(self, data, model):
        # unpack model
        self.data = data   
        (self.very_pos_features, self.pos_features,
         self.very_neg_features, self.neg_features, self.neu_features,
         self.very_pos_prior, self.pos_prior,
         self.neg_prior, self.very_nag_prior, self.neu_prior, 
         self.total_num_feature,
         self.num_very_pos_feature ,
                  self.num_pos_feature ,
                  self.num_very_neg_feature,
                  self.num_neg_feature ,
                  self.num_neu_feature )  = model
                 
    def classify(self):
        results = []
        #initialise score to their prior probability
        
        #calculate each class probability
        for sentences in self.data['Phrase']:
            very_pos_score = self.very_pos_prior
            pos_score = self.pos_prior
            neg_score = self.neg_prior 
            very_neg_score = self.very_nag_prior
            neu_score = self.neu_prior 
            for words in sentences:
                if words in self.very_pos_features:
                    very_pos_score *= self.very_pos_features[words]
                    pos_score *= self.pos_features[words]
                    very_neg_score *= self.very_neg_features[words]
                    neg_score *= self.neg_features[words]
                    neu_score *= self.neu_features[words]
                
                else:
                    very_pos_score *= 1/(self.total_num_feature + self.num_very_pos_feature)         
                    pos_score *= 1/(self.total_num_feature + self.num_pos_feature)
                    very_neg_score *= 1/(self.total_num_feature + self.num_very_neg_feature)
                    neg_score *= 1/(self.total_num_feature + self.num_neg_feature)
                    neu_score *= 1/(self.total_num_feature + self.num_neu_feature)
            # write guess class into result    
            if max(very_pos_score, pos_score,neg_score,very_neg_score,neu_score) == very_pos_score:
                results.append(4)
            elif max(very_pos_score, pos_score,neg_score,very_neg_score,neu_score) == pos_score:
                results.append(3)    
            elif max(very_pos_score, pos_score,neg_score,very_neg_score,neu_score) == very_neg_score:
                results.append(0)
            elif max(very_pos_score, pos_score,neg_score,very_neg_score,neu_score) == neg_score:
                results.append(1)
            elif max(very_pos_score, pos_score,neg_score,very_neg_score,neu_score) == neu_score:
                results.append(2)
            else:
                continue
        #compare guess with ground truth to get percent of accuracy
        self.data['Results'] = results
        comparison_column = np.where(self.data['Sentiment'] == self.data['Results'], 1, 0)
        accuracy = (sum(comparison_column))/(self.data.shape[0])
        #self.data['Compare'] = comparison_column 
        return accuracy         
    
#==============================================================================
# MAIN

# load data
train_data = FileLoader('train.tsv')   
# select features from preprocessed data
featured_data = FeatureSelection(train_data.preprocess())
#print(featured_data.bigram())
# use features to train model
train = Train(featured_data.bag_of_words())
train.priorProbability()
train.likelyhood()
#very_pos_features, pos_features, very_neg_features,neg_features, neu_features = train.likelyhood()
test_data = FileLoader('dev.tsv')
featured_test_data = FeatureSelection(test_data.preprocess())
#print(featured_test_data.bigram())
model = train.get_model()
classifier = BayesClassifier(featured_test_data.bag_of_words(),model)
print(classifier.classify())

       
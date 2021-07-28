# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:50:46 2020

@author: wenjie zhang
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
import classifier

# plot confusion matrix 
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum()

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
    
#initialise matrix
cm5 = np.zeros((5,5))
cm3 = np.zeros((3,3))
#compute each cell value for 3 value sentiment
for num, row in classifier.all_word_test1[['Sentiment', 'myClassification3']].iterrows():
    cm3[(row['Sentiment'])][row['myClassification3']] += 1
#compute each cell value for 5 value sentiment
for num, row in classifier.all_word_test[['Sentiment', 'myClassification5']].iterrows():
    cm5[row['Sentiment']][row['myClassification5']] +=1   
#plot matrix    
plot_confusion_matrix(cm=cm5, target_names=classifier.sentiment_value5, normalize=False)
plot_confusion_matrix(cm=cm3, target_names=classifier.sentiment_value3, normalize=False)

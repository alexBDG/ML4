# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:42:40 2019

@author: Alexandre
"""

import numpy as np
from nltk.tokenize import word_tokenize
import codecs


def sentencize(data):
    N = len(data)
    sentences_A = []
    sentences_B = []
    relatidness_score = np.zeros(N)
    
    for i in range(N):
        sentences_A += [data.loc[i,"sentence_A"]]
        sentences_B += [data.loc[i,"sentence_B"]]
        relatidness_score[i] = data.loc[i,"relatedness_score"]
        
    return (sentences_A+sentences_B,relatidness_score)




def load_sts(dsfile, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                if skip_unlabeled:
                    continue
                else:
                    labels.append(-1.)
            else:
                labels.append(float(label))
            s0.append(word_tokenize(s0x))
            s1.append(word_tokenize(s1x))
    return (s0+s1, np.array(labels))




def load_sick2014(dsfile, mode='relatedness'):
    """ load a dataset in the sick2014 tsv .txt format;

    mode='relatedness': use the sts relatedness score as label
    mode='entailment': use -1 (contr.), 0 (neutral), 1 (ent.) as label """
    rte_lmappings = {'contradiction': np.array([1,0,0]), 'neutral': np.array([0,1,0]), 'entailment': np.array([0,0,1])}

    s0 = []
    s1 = []
    labels = []
    with open(dsfile) as f:
        first = True
        for line in f:
            if first:
                # skip first line with header
                first = False
                continue
            line = line.rstrip()
            pair_ID, sentence_A, sentence_B, relatedness_score, entailment_judgement = line.split('\t')
            if mode == 'relatedness':
                label = float(relatedness_score)
            elif mode == 'entailment':
                if entailment_judgement.lower() in rte_lmappings:
                    label = rte_lmappings[entailment_judgement.lower()]
                else:
                    raise ValueError('invalid label on line: %s' % (line,))
            else:
                raise ValueError('invalid mode: %s' % (mode,))
            labels.append(label)
            s0.append(word_tokenize(sentence_A))
            s1.append(word_tokenize(sentence_B))
    return (s0+s1, np.array(labels))




def cosine(x,y):
    cosi = np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))
    if cosi < 0:
        print("neg !")
    return cosi




def error_detector(V_sentence):
    ERROR_index = []
    for k in range(len(V_sentence)):
        for i in range(len(V_sentence[0])):
            x = V_sentence[k,i]
            if (x is np.nan) or (x != x):
                ERROR_index += [(k,i)]
    return ERROR_index



def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    if not str:
        print('The text to be tokenized is not a string. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)

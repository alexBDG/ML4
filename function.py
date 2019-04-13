# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:42:40 2019

@author: Alexandre
"""

import numpy as np
from nltk.tokenize import word_tokenize
import codecs
import pandas as pd
import csv



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



def load_PSL():
    word_embedding = {}
    with codecs.open('word_embedding\\paragram_300_sl999.txt', 'rb', encoding="utf-8", errors='replace') as f:
        for line in f:
            try:
                line_vec = line.rstrip().split(' ')
            except:
                line.encode('utf-8').strip()
                line_vec = line.rstrip().split(' ')
            word_embedding[line_vec[0]] = np.array(line_vec[1:],dtype=float)
    return word_embedding



def load_GloVe(version="6B.300d"):
#    word_embedding = {}
    word_embedding = pd.read_table("word_embedding\\glove.{0}.txt".format(version), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
#    for i in range(len(words)):
#        word = words.index[i]
#        word_embedding[word] = words.loc[word].values
    return word_embedding



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

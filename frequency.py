# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:31:04 2019

@author: Alexandre
"""


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import json
import pandas as pd


sent = 'This is an example sentence for this'

def create_dictionnary(sent):
    fdist = FreqDist()
    for word in word_tokenize(sent):
        fdist[word.lower()] += 1
    print(fdist.freq("this"))
        
    pdist = {}
    for word in fdist:
        pdist[word] = fdist.freq(word)
        
    return pdist
    
    
# To have the frequency of "word" ---> fdist.freq('word')
    

def read_dictionnary(file_name):
    fdist = pd.read_csv('enwiki-20190320-words-frequency.txt', sep=" ",header=None)
    pdist = {}
    sume = 0
    for i in range(len(fdist)):
        sume += fdist.loc[i,1]
    for i in range(len(fdist)):
        pdist[fdist.loc[i,0]] = fdist.loc[i,1]/sume
    return pdist
    


    
def save_file(pdist,data_name):
    json_file = json.dumps(pdist)
    with open("word_frequency\{0}.json".format(data_name),"w") as f:
        f.write(json_file)
    
    
    
def load_file(data_name):
    with open("word_frequency\{0}.json".format(data_name)) as f:
        pdist = json.load(f)
    return pdist


#pdist = read_dictionnary(sent)
#save_file(pdist,"enwiki_2184780")
#pdist_test = load_file("enwiki_2184780")

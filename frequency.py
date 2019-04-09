# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:31:04 2019

@author: Alexandre
"""


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import json


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
    


    
def save_file(pdist,data_name):
    json_file = json.dumps(pdist)
    with open("word_frequency\{0}.json".format(data_name),"w") as f:
        f.write(json_file)
    
    
    
def load_file(data_name):
    with open("word_frequency\{0}.json".format(data_name)) as f:
        pdist = json.load(f)
    return pdist


pdist = create_dictionnary(sent)
save_file(pdist,"test")
pdist_test = load_file("test")
print(pdist)
print(pdist_test)

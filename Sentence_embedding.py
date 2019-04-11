# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:40:15 2019

@author: Alexandre
"""


import numpy as np
import pandas as pd
import csv
import frequency
import time
import function
import sys
import os

    
        

def Algo(a=1e-3,task="SICK 2014",methode="WR"):
    ini = time.time()
    path = os.getcwd()
    
    if task == "STS 2012":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2012\MSRpar.test.tsv"
        task_family = "STS"
    if task == "STS 2013":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2013\headlines.test.tsv"
        task_family = "STS"
    if task == "STS 2014":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2014\headlines.test.tsv"
        task_family = "STS"
    if task == "STS 2015":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2015\headlines.test.tsv"
        task_family = "STS"
    if task == "STS 2016":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2016\headlines.test.tsv"
        task_family = "STS"
    if task == "SICK 2014":
        file_name = path + r"\dataset-sts-master\data\sts\sick2014\SICK_test_annotated.txt"
        task_family = "SICK"
    if task == "Twitter 2015":
        file_name = path + r"\dataset-sts-master\data\sts\semeval-sts\2012\MSRpar.test.tsv"
        task_family = "Twitter"
    
    start = time.time()
    if task_family == "STS":
        (sentences,known_score) = function.load_sts(file_name)
    elif task_family == "SICK":
        (sentences,known_score) = function.load_sick2014(file_name)
    else:
        print("NOT DEFINED !")
        assert("error")
    print("############################################")
    print("sentence are created from {0}, this took {1} seconds".format(task,round(time.time()-start,3)))
    
    start = time.time()
    pdist = frequency.load_file("enwiki_2184780")
    print("############################################")
    print("enwiki is loaded for the probas, this took {0} seconds".format(round(time.time()-start,3)))
    start = time.time()
    words = pd.read_table("word_embedding\glove.6B.300d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    print("############################################")
    print("glove.42B.300d is loaded for the enbedding words, this took {0} seconds".format(round(time.time()-start,3)))
    N = len(known_score)
    
    start = time.time()
    unknown_words = {}
    unknown_probas = {}
    V_sentence = np.zeros((2*N,words.shape[1]))
    i=0
    for s in sentences:
        ph = "\rProgression: {0} % ".format(round(float(100*i)/float(2*N-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        sume = 0
        s_tolk = [word.lower() for word in s if word.isalpha()]
        for w in s_tolk:
            try:
                try:
                    p = pdist[w]
                except:
                    if w == 'a':
                        w = 'an'
                        p = pdist[w]
                    else:
                        p = 0.
                        try:
                            unknown_probas[w] += 1
                        except:
                            unknown_probas[w] = 1
                if methode == "avg":
                    sume += words.loc[w].values
                elif methode == "WR":
                    sume += a/(a+p)*words.loc[w].values
            except:
                try:
                    unknown_words[w] += 1
                except:
                    unknown_words[w] = 1
        V_sentence[i] = sume/len(s_tolk)
        i+=1
    print("############################################")
    print("sentence vectors are created, this took {0} seconds".format(round(time.time()-start,3)))
    print()
    print("This words are not in the <embedding words> database :\n",unknown_words)
    print()
    print("This words are not in the <probas words> database :\n",unknown_probas)
    print()
    
    start = time.time()
    uh, Sigma, vh = np.linalg.svd(V_sentence.transpose(), full_matrices=True)
    print("############################################")
    print("singular vector are found, this took {0} seconds".format(round(time.time()-start,3)))

        
    start = time.time()
    u = np.zeros(uh.shape[0])
    u[0] = Sigma[0]
    v = np.copy(u)
    v.shape = (np.size(u),1)
    u.shape = (1,np.size(u))

    for s in range(2*N):
#        V_sentence[s] -= np.dot( np.dot(v, u) , V_sentence[s]) #22.78111123433595
        V_sentence[s] -= Sigma[0]*Sigma[0]*V_sentence[s] #64.99764426105642
    print("############################################")
    print("sentence vectors are updated, this took {0} seconds".format(round(time.time()-start,3)))

        
    start = time.time()
    unknown_score = np.zeros(N)
    scores = np.zeros((N,2))
    for i in range(N):
        unknown_score[i] = 5*function.cosine(V_sentence[i],V_sentence[i+N])
        scores[i] = [unknown_score[i],known_score[i]]
    
    Pearson_s_matrix = np.corrcoef(unknown_score,known_score)
    Pearson_s_coef = Pearson_s_matrix[0,1]
    print("############################################")
    print("pearson's coefficent is calculated, this took {0} seconds".format(round(time.time()-start,3)))
    
    
    print("############################################")
    delta_t = round(time.time()-ini,3)
    print("End, this took {0} seconds".format(delta_t))
    
    return (Pearson_s_coef,delta_t,V_sentence,scores)



#DONNER LE V_sentence DU AVERAGE GloVe =  Moy des embbeding vector des mots

def run():
    
    (Pearson_s_coef,delta_t,V_sentence,scores) = Algo(1e-3)
    print("For the STS12_test, we have [Pearson_s_coef x 100] = ",Pearson_s_coef*100)
    
    ERROR_index = function.error_detector(V_sentence)
    print("These index have nan value : \n",ERROR_index)
    


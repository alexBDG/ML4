# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:45:02 2019

@author: Alexandre
"""


import numpy as np
import pandas as pd
import csv
import frequency
from nltk.tokenize import word_tokenize





def find_frequence(data):
    words = {}
    for data_point in data:
        for word in data_point.split(' '):
            if (word.lower()) in words:
                words[word.lower()] += 1
            else:
                words[word.lower()] = 1
    try:
        words.pop('')
    except:
        print("no --> '' in the list")
  
    n = len(words)
#    P_word = np.zeros(n)
    P_word = {}
    V_word = []
    word
    i=0
    for word in words:
#        P_word[i] = words.get(word)/n
        P_word[word] = words.get(word)/n
        V_word += [word]
        i+=1
    V_word = np.array(V_word)
        
    return(V_word,P_word)
    
    

data = ['This mini-project is due on April 12th at 11:59pm. Late submissions will be accepted without penalty until April 15th at 11:59pm, but no submissions will be accepted after that date.',
        'This mini-project is to completed in groups of three. All members of a group will recieve the same grade. It is not expected that all team members will contribute equally to all components. However every team member should make integral contributions to the project.',
        'You will submit your assignment on MyCourses as a group. As with previous mini-projects, you must register your group on MyCourses and any group member can submit.',
        'You are free to use any programming language or toolbox you want with this project.']

data = pd.read_csv('SICK_test.txt', sep="\t")




def sentencize0(data):
    sentence = []
    if len(data) == 1:
        for dat in data.split('. '):
            sentence += dat.split('.')
    else:
        for dat in ' '.join(data).split('. '):
            sentence += dat.split('.')
    try:
        sentence.remove('')
        return sentence
    except:
        return sentence
    

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


def cosine(x,y):
    cosi = np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))
    if cosi < 0:
        print("neg !")
    return cosi
        


#sentences = np.array(['i am','you are','he is','i will yes']) #set of sentences S
#V_word = np.array(['I','am','you','are','he','is','will','yes'])
#word_embedding = np.array([])
#proba_word = np.array([0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1])


def word_embedder(V_word):
    words = pd.read_table("glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    word_embedding = np.zeros((len(V_word),50))
    i=0
    for word in V_word:
        try:
            word_embedding[i] = words.loc[word].as_matrix()
        except:
            print("eroor?")
        i+=1
    
    return word_embedding





def Algo2(a,data):
    
    (sentences,known_score) = sentencize(data)
    pdist = frequency.load_file("enwiki_2184780")
    words = pd.read_table("glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    N = len(known_score)
    
    errors = []
    unknown_words = {}
    V_sentence = np.zeros((2*N,50))
    i=0
    for s in sentences:
        sume = 0
        s_tolk = word_tokenize(s)
        s_tolk = [word.lower() for word in s_tolk if word.isalpha()]
        for w in s_tolk:
            try:
                try:
                    p = pdist[w]
                except:
                    if w == 'a':
                        w = 'an'
                        p = pdist[w]
                    else:
                        print("Unknown proba : ",w)
                        p = 0.
                sume += a/(a+p)*words.loc[w].as_matrix()
            except:
                print("Unknown word : ",w)
                print("Unknown sentence : ",s_tolk,s)
                unknown_words[w] += 1
                errors += [i]
        V_sentence[i] = sume/len(s_tolk)
        i+=1
        
    print("This words are not in the <embedding words> database :",unknown_words)
#    print("Errors with index :",[i for i in errors])
#    for i in range(len(errors)):
#        print(V_sentence[i])
        
#    V_sentence = np.array(V_sentence)
#    V_sentence_t = V_sentence.transpose()
    
    print(V_sentence.shape)
    print(V_sentence.transpose().shape)
    uh, Sigma, vh = np.linalg.svd(V_sentence.transpose(), full_matrices=True)
    print(uh.shape,Sigma.shape,vh.shape)

        
    u = np.zeros(uh.shape[0])
    u[0] = Sigma[0]
    v = np.copy(u)
    v.shape = (np.size(u),1)
    u.shape = (1,np.size(u))
#    print(v)
#    print(u)
#    print(np.dot(v, u))
    
    for s in range(2*N):
#        V_sentence[s] -= np.dot( np.dot(v, u) , V_sentence[s])
        V_sentence[s] -= Sigma[0]*Sigma[0]*V_sentence[s]

        
    
    unknown_score = np.zeros(N)
    scores = np.zeros((N,2))
    for i in range(N):
        unknown_score[i] = 5*cosine(V_sentence[i],V_sentence[i+N])
        scores[i] = [unknown_score[i],known_score[i]]
    
    Pearson_s_matrix = np.corrcoef(unknown_score,known_score)
    Pearson_s_coef = Pearson_s_matrix[0,1]
    
    return (Pearson_s_coef,V_sentence,scores)



#DONNER LE V_sentence DU AVERAGE GloVe =  Moy des embbeding vector des mots

(Pearson_s_coef,V_sentence,scores) = Algo2(1e-4,data)
print("For the SICK_test, we have [Pearson_s_coef x 100] = ",Pearson_s_coef*100)


def error_detector(V_sentence):
    ERROR_index = []
    for k in range(len(V_sentence)):
        for i in range(len(V_sentence[0])):
            x = V_sentence[k,i]
            if (x is np.nan) or (x != x):
                ERROR_index += [(k,i)]
    return ERROR_index

ERROR_index = error_detector(V_sentence)
print("These index have nan value : \n",ERROR_index)



def Algo1(a,data):
    
    sentences = sentencize(data)
    (V_word,P_word) = find_frequence(sentences)
#    print(V_word)
#    print(P_word)
#    print(sentences)
#    word_embedding = word_embedder(V_word)
    words = pd.read_table("glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    
#    V_sentence = np.zeros(len(sentences))
    V_sentence = []
    i=0
    for s in sentences:
        sume = 0
        for w in s.split(" "):
            w = w.lower()
            try:
#                sume += a/(a+P_word[w])*word_embedding[w]
                sume += a/(a+P_word.get(w))*words.loc[w].as_matrix()
            except:
                print("Unknown word : ",w)
#        V_sentence[i] = sume/len(s)  # len ???
        V_sentence += [sume/len(s)]  # len ???
        i+=1
        
    X = np.array(V_sentence)
    X.transpose()
    
    uh, Sigma, vh = np.linalg.svd(X, full_matrices=True)
    
    print(uh.shape,Sigma.shape,vh.shape)
    
    u = np.zeros(vh.shape[0])
    u[0] = Sigma[0]
    
    for s in range(len(sentences)):
        V_sentence[s] -= np.dot(u, np.dot(u.transpose() , V_sentence[s]))
    
    return V_sentence


#V_sentence = Algo1(1e-4,data)

def evaluate(V_sentence):
    
    np.corrcoef(x,y)

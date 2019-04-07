# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:45:02 2019

@author: Alexandre
"""


import numpy as np

def prob(w):
    return 0.1

def Pr_known_cs(w,s,alpha,beta,c,v):
    p = prob(w)
    c_t =beta*c[0] + (1-beta)*c[s]
    Pr = alpha * p + (1-alpha) * np.exp(np.dot(c_t,v[w])) / Z
    return Pr




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
    P_word = np.zeros(n)
    V_word = []
    word
    i=0
    for word in words:
        P_word[i] = words.get(word)/n
        V_word += [word]
        i+=1
    V_word = np.array(V_word)
        
    return(V_word,P_word)
    
    

data = ['This mini-project is due on April 12th at 11:59pm. Late submissions will be accepted without penalty until April 15th at 11:59pm, but no submissions will be accepted after that date.',
        'This mini-project is to completed in groups of three. All members of a group will recieve the same grade. It is not expected that all team members will contribute equally to all components. However every team member should make integral contributions to the project.',
        'You will submit your assignment on MyCourses as a group. As with previous mini-projects, you must register your group on MyCourses and any group member can submit.',
        'You are free to use any programming language or toolbox you want with this project.']

def sentencize(data):
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


#sentences = np.array(['i am','you are','he is','i will yes']) #set of sentences S
#V_word = np.array(['I','am','you','are','he','is','will','yes'])
#word_embedding = np.array([])
#proba_word = np.array([0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1])





def Algo1(a,data,word_embedding):
    
    sentences = sentencize(data)
    (V_word,P_word) = find_frequence(sentences)
    
    V_sentence = np.zeros(len(sentences))
    for s in len(sentences):
        sume = 0
        for w in s:
            sume += a/(a+P_word[w])*word_embedding[w]
        V_sentence[s] = sume/len(s)  # len ???
        
    X = np.array(V_sentence)
    X.transpose()
    
    u, Sigma, vh = np.linalg.svd(X, full_matrices=True)
    
    u = Sigma[:,0]
    for s in len(sentences):
        V_sentence[s] -= np.dot(u, np.dot(u.transpose() , V_sentence[s]))
    
    return V_sentence
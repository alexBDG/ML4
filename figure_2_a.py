# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:54:01 2019

@author: Alexandre
"""

import Sentence_embedding



curve = "GloVe+avg"

if curve == "GloVe+WR":
    with open("figure_2_a\{0}.txt".format(curve),"w") as file:
        file.write("a Pearson\n")
        for i in [1,2,3,4,5]:
            a = 10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            a = 3*10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "GloVe+avg":
    with open("figure_2_a\{0}.txt".format(curve),"w") as file:
        file.write("a Pearson\n")
        (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(1,"STS 2012","avg")
        file.write("{0} {1}\n".format(a,Pearson_s_coef))
        for i in [1,2,3,4,5]:
            a = 10**(-i)
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            a = 3*10**(-i)
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
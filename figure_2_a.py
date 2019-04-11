# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:54:01 2019

@author: Alexandre
"""

import Sentence_embedding



curve = "GloVe_WR"

if curve == "GloVe_WR":
    with open("figure_2_a\GloVe+WR.txt","w") as file:
        file.write("a Pearson\n")
        for i in [1,2,3,4,5]:
            a = 10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            a = 3*10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
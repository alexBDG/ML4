# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:54:01 2019

@author: Alexandre
"""

import Sentence_embedding
import pandas as pd
import matplotlib.pyplot as plt
import os




if not os.path.exists(os.getcwd()+"\\figure_2_a"):
    os.makedirs("figure_2_a")
    

curve = "PSL"

if curve == "GloVe+WR":
    with open("figure_2_a\{0}.txt".format(curve),"w") as file:
        file.write("a Pearson\n")
        for i in [1,2,3,4,5]:
            a = 3*10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            a = 10**(-i)
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR")
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "GloVe":
    with open("figure_2_a\{0}.txt".format(curve),"w") as file:
        file.write("a Pearson\n")
        (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(1,"STS 2012","avg")
        for i in [1,2,3,4,5]:
            a = 3*10**(-i)
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            a = 10**(-i)
            file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
            
            
            
def plot_fig2a():
    plots = [["PSL+WR","blue",""],
             ["PSL","blue","dashed"],
             ["GloVe+WR","green","solid"],
             ["GloVe","green","dashed"],
             ["SN+WR","orange","solid"],
             ["SN","orange","dashed"]]
    
    plt.figure()
    
    for curve in plots:
        try:
            curve_data = pd.read_csv(r"figure_2_a\{0}.txt".format(curve[0]), sep=" ",header=0)
            plt.semilogx([a for a in curve_data["a"]],[p for p in curve_data["Pearson"]],linestyle=curve[2],color=curve[1],label=curve[0])
        except:
            print("<{0}.txt> is not in the directory figure_2_a !!".format(curve[0]))
    
    plt.xlabel("Weighting parameter a")
    plt.ylabel("Pearson's coefficient")
    plt.legend(loc='lower right',ncol=2)
    plt.xlim(1e-5,1e0)
    plt.tick_params(top=1,right=1,direction='in',which='both')
    plt.savefig(r"figure_2_a\figure_2_a.png")
    plt.show()
    plt.close()
    

            
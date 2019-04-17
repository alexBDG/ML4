# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:54:01 2019

@author: Alexandre
"""

import Sentence_embedding
import pandas as pd
import matplotlib.pyplot as plt
import os




if not os.path.exists(os.getcwd()+"/figure_2_a"):
    os.makedirs("figure_2_a")
    

curve = "GloVe+h"

if curve == "GloVe+WR":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "GloVe":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(1,"STS 2012","avg","GloVe")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "PSL+WR":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR","PSL")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","WR","PSL")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "PSL":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(1,"STS 2012","avg","PSL")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "GloVe+bin":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","bin","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","bin","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
            
elif curve == "GloVe+h":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","h","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","h","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                
elif curve == "GloVe+g":
    if not os.path.exists("figure_2_a/{0}.txt".format(curve)):
        with open("figure_2_a/{0}.txt".format(curve),"w") as file:
            file.write("a Pearson\n")
            for i in [1,2,3,4,5]:
                a = 3*10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","g","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))
                a = 10**(-i)
                (Pearson_s_coef,delta_t,V_sentence,scores) = Sentence_embedding.Algo(a,"STS 2012","g","GloVe")
                file.write("{0} {1}\n".format(a,Pearson_s_coef))

            
            
def plot_fig2a():
    plots = [["PSL+WR","blue","solid"],
             ["GloVe+WR","green","solid"],
             ["GloVe+h","red","dashdot"],
             ["GloVe+bin","red","dashdot"],
             ["PSL","blue","dashed"],
             ["GloVe","green","dashed"],
             ["GloVe+g","red","dashdot"]]
    
    plt.figure()
    
    for curve in plots:
        try:
            curve_data = pd.read_csv(r"figure_2_a/{0}.txt".format(curve[0]), sep=" ",header=0)
            plt.semilogx([a for a in curve_data["a"]],[p for p in curve_data["Pearson"]],linestyle=curve[2],color=curve[1],label=curve[0])
        except:
            print("<{0}.txt> is not in the folder figure_2_a !!".format(curve[0]))
    
    plt.xlabel("Weighting parameter a")
    plt.ylabel("Pearson's coefficient")
    plt.legend(loc='lower right',ncol=2)
    plt.xlim(1e-5,1e0)
    plt.tick_params(top=1,right=1,direction='in',which='both')
    plt.savefig(r"figure_2_a\figure_2_a.png")
    plt.show()
    plt.close()
    

            
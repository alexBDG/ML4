# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:51:22 2019

@author: Alexandre
"""

import os
import sys
import frequency




def check_embedding():
    PSL = "https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing"
    GloVe = "http://nlp.stanford.edu/data/glove.6B.zip"
    dataset = [[PSL,'paragram_300_sl999.txt'],[GloVe,'glove.6B.300d.txt']]
    for link,file in dataset:
        check(link,file)


def check(link,file):
    directory = os.getcwd() + "\\word_embedding"
    if not os.path.exists(directory):
        os.makedirs("word_embedding")
        sys.exit('\nGo to :\n{0}\nDownload and unzip it, take the file :\n{2}\nPlace it in the directory :\n{1}\n'.format(link,directory,file))
    else:
        if not os.path.exists(directory+'\\'+file):
            sys.exit('\nGo to :\n{0}\nDownload and unzip it, take the file :\n{2}\nPlace it in the directory :\n{1}\n'.format(link,directory,file))
        
        
def check_sts():
    link = "https://github.com/brmson/dataset-sts/tree/master/data/sts"
    directory = os.getcwd() + "\\sts"
    if not os.path.exists(directory):
        sys.exit('\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n'.format(link,os.getcwd()))
    else:
        if not os.path.exists(directory+"\\semeval-sts\\2012\\MSRpar.test.tsv"):
            sys.exit("\n<semeval-sts\\2012\MSRpar.test.tsv> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))
        elif not os.path.exists(directory+"\\semeval-sts\\2013\\headlines.test.tsv"):
            sys.exit("\n<semeval-sts\\2013\headlines.test.tsv> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))
        elif not os.path.exists(directory+"\\semeval-sts\\2014\\headlines.test.tsv"):
            sys.exit("\n<semeval-sts\\2014\\headlines.test.tsv> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))
        elif not os.path.exists(directory+"\\semeval-sts\\2015\\headlines.test.tsv"):
            sys.exit("\n<semeval-sts\\2015\\headlines.test.tsv> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))
        elif not os.path.exists(directory+"\\semeval-sts\\2016\\headlines.test.tsv"):
            sys.exit("\n<semeval-sts\\2016\\headlines.test.tsv> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))
        elif not os.path.exists(directory+"\\sick2014\\SICK_test_annotated.txt"):
            sys.exit("\n<sick2014\\SICK_test_annotated.txt> is missing in {0} !\nGo to :\n{0}\nDownload all the folder <sts>, place it in the current directory :\n{1}\n".format(directory,link,os.getcwd()))

         
            
def check_frequency():
    link = "https://github.com/IlyaSemenov/wikipedia-word-frequency/tree/master/results"
    file = "enwiki-20190320-words-frequency.txt"
    directory = os.getcwd() + "\\word_frequency"
    if not os.path.exists(directory):
        os.makedirs("word_frequency")
        sys.exit('\nGo to :\n{0}\nDownload the file :\n{1}\nPlace it in the folder :\n{2}\n'.format(link,file,directory))
    else:
        if not os.path.exists(directory+"\\enwiki_2184780.json"):
            pdist = frequency.read_dictionnary()
            frequency.save_file(pdist,"enwiki_2184780")
    
    
    
def check_env():
    check_embedding()
    check_frequency()
    check_sts()
    print("Environment OK")

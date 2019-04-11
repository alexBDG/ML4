# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:36:44 2019

@author: Alexandre
"""



file_name = "word_frequency/wikipedia-word-frequency-master/result/enwiki-20190320-words-frequency.txt"




with open(file_name, "rb") as file:
    enwiki_compressed = file.read()
enwiki = dec.decompress(enwiki_compressed)

print(enwiki.shape)



# ML4
Mini-Projet 4 - Task 1


Alexandre Banon; Vincent Delmas; Michael Haaf



## Paper :

Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017. A simple but tough-to-beat baseline for sentence embed-dings. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings.



## Files :

<Sentence_embedding.py> contains our reproduction of the algorithm 1 from the paper (Arora et al.,2017).

<function.py> and <frequence.py> contain useful functions for the main algorithm.

<figure_2_a.py> contains a script to produce the figure 2 of our report.

<environment.py> check if the work space is well organized, and shares the link to dowload the datasets.



Python libraries :

numpy, time, sys, os, math, nltk, codecs, pandas, csv, matplotlib



Link for Dowloading the data :

STS (2012 to 2015) and SICK 2014 on the following Github : https://github.com/brmson/dataset-sts/tree/master/data/sts

GloVe embedding pre-trained vectors : http://nlp.stanford.edu/data/glove.6B.zip

PSL embedding pre-trained vectors : https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing

enwiki database from Wikipedia articles : https://github.com/IlyaSemenov/wikipedia-word-frequency/tree/master/results



Parameters of the function <Algo> :
  
<a>
A number, parameter of the weight factor of the method proposed by the paper

<task>
Possible values :   "STS 2012", "STS 2013", "STS 2014", "STS 2015", "SICK 2014"

<methode>
Possible values :   "WR", "avg", "bin", "g", "h"

<word_embedding>
Possible values :   "GloVe", "PSL"




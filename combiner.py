# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:24:50 2019

@author: Amin

This is a simple file developing a list of 
"""

import glob  # to get a list of current files
import numpy as np  # nump package

def combiner(folder):
    
    read_files = glob.glob(folder + r"/*.txt") # reading all the files into one list
    # clean later the following line     
    #  the following piece of code has two functions
    #  appends end of line word to each sentence
    #  get a list of length to be used for choosing window size for LSTM
    cache = {}  # a dictionary developed to keep the analytics
    corpus_list = []  # corpus in a string 
    num_words = [] # empty array to save number of words
    num_chars = [] # empty array to save number of characters
    for f in read_files:
        with open(f, "r", encoding="utf8") as infile:
            for sentence in infile:
                if not sentence.strip(): continue  # skipping over empt lines
                sentence = sentence.strip()
                corpus_list.append(sentence)  # adding the current sentence to the corpus
                list_of_words = sentence.split(" ")  # list of words in sentence
                num_words.append(len(list_of_words))  # number of words in list
                num_chars.append(len(sentence))  # umber of characters in sentence
    # we first delete lines with small number of words
    corpus_list = [corpus_list[i] for i, n in enumerate(num_words) if n > 1]          
    corpus = " . ".join(corpus_list)            
    #  analytis on the poem length
    mean_length = np.mean(num_chars)  # mean of number of words   
    median_length = np.median(num_chars)  # median of the number of words
    #  developing the cache
    cache['median'] = median_length  # median of the sentence
    cache['mean'] = mean_length  # mean of the sentence
    
    return corpus, cache,  median_length
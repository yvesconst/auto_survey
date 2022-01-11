#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 15:52:31 2022

@author: ytadjota
"""

from datasets import load_dataset
from bertopic import BERTopic
import pandas as pd 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import spacy



class TopicModel(BERTopic):
    def __init__(self,
                 language: str="english",
                 min_topic_size: int=20,
                 verbose: bool=True,
                 nr_topics: Union[int, str]=None):
        pass
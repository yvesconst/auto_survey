#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:05:38 2022

@author: ytadjota
"""

import pickle
import numpy as np
import tensorflow as tf
from transormer import pipeline
import torch as T

class UserModel():
    def __init__(self, params):
        feature_size = params["sentence_size"]+params["embedding_size"]
        self.question = tf.placeholder(shape=[None, params["question"]],dtype=tf.float32)
        self.label_engagement = tf.placeholder(shape=[None],dtype=tf.int32)
        self.user = tf.placeholder(shape=[None],dtype=tf.int32)
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
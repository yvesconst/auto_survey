#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 04:34:43 2022

@author: ytadjota
"""

from pdf2image import convert_from_path
import os

doc_name = '1'

try:
    images = convert_from_path('./1.pdf')
    relative_path = './images/{0}'.format(doc_name)
    os.makedirs(relative_path)
    for count, img in enumerate(images):
        
        img.save(relative_path +'/{0}.jpg'.format(str(count)), 'JPEG')
except:
    print("Error")
 
 
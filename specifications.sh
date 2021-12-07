#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Nov  7 14:11:39 2021

@author: reem
"""

python scripts/main.py \
    --technique='baseline RESNET' \
    --RGBpath= 'ML Pro Dataset RGB' \
    --GREYpath= 'ML Pro Dataset grayscale' \
    --batchsize=16 \
    --epochs=25 \
    --histogram='false' \
    --showtransformed = 'false'
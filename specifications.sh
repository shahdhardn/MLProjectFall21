#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Nov  7 14:11:39 2021

@author: reem
"""

python scripts/main.py \
    --technique=hg2 \
    --RGBpath= 'ML Pro Dataset RGB' \
    --GREYpath= 'ML Pro Dataset grayscale' \
    --batchsize=16 \
    --epochs=35 \
    --train-batch=24 \
    --workers=24 \
    --test-batch=24 \
    --lr=1e-3 \
    --schedule 15 17
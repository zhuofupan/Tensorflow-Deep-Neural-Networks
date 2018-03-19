# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:53:17 2018

@author: Administrator
"""
import numpy as np
for i in range(3):
    print(np.fromfunction(lambda j,k:(i==k).astype(int),(4,5)))
    #print(np.full(4,i).reshape(4,1))
    print()


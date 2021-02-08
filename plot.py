# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:26:06 2021

@author: intisar
"""



import matplotlib.pyplot as plt
import json
import numpy as np

with open('./checkpoint/w_o_d_subset_sampler_5experts/exp_subset_sampler.json', encoding='utf-8') as fl:
    b = fl.read()
   # print(b['text'])
print (b)

a = [1760, 1780, 1784, 1795, 1802, 1790, 1799, 1789, 1784, 1793, 1803, 1794, 1800, 1809, 1808, 1812, \
1801, 1809, 1803, 1811, 1811, 1810, 1810, 1803, 1802, 1802, 1795, 1803, 1809, 1799]
c = [8578, 8403, 8358, 8206, 8010, 8128, 8042, 7975, 8063, 7927, 8014, 7759, 7815, 7694, 7670, 7720, 7596, 7598, 7616, 7496, 7479, 7510, 7396, 7438, 7442, 7407, 7341, 7256, 7276, 7206]
a = [num*100/2000 for num in a] 
#plt.plot()
c = [num*100/10000 for num in c] 
b = np.arange(1, 31)  



fig, ax = plt.subplots()

ax.plot(b,a)
ax.plot(b,c)

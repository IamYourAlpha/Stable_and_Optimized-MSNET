# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:26:06 2021

@author: intisar
"""



import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import pandas as pd



b = np.arange(1, 31)  


acc = pd.read_csv('./checkpoint/w_o_d_subset_sampler_5experts/exp_subset_sampler.csv')
acc = pd.read_csv('out_30_finetuned.csv')

#
#sns.relplot(
#    data=acc,
#    x="epochs", y="3_5_test_experts",
#    kind="line"
#)
print (acc.describe())
a_3_5_test_subset = np.array(acc['3_5_test_experts'])
a_3_5_test_all = np.array(acc['3_5_test_all'])

fig, ax = plt.subplots()


a_3_5_test_subset = np.array(acc['3_5_test_experts'])
a_3_5_test_subset = [num/2000*100 for num in a_3_5_test_subset]

a_3_5_test_all = np.array(acc['3_5_test_all'])
a_3_5_test_all = [num/10000*100 for num in a_3_5_test_all]


ax.plot(b,a_3_5_test_subset, label='3-5 class accuracy')
ax.plot(b,a_3_5_test_all, label='all class accuracy')
legend = ax.legend(loc='upper centre', shadow=False, fontsize='x-large')
plt.grid(True)














a_4_7_test_subset = np.array(acc['2_3_test_experts'])
a_4_7_test_subset = [num/2000*100 for num in a_4_7_test_subset]

a_4_7_test_all = np.array(acc['2_3_test_all'])
a_4_7_test_all = [num/10000*100 for num in a_4_7_test_all]


fig, ax = plt.subplots()
ax.plot(b,a_4_7_test_subset, label='2-3 class accuracy')
ax.plot(b,a_4_7_test_all, label='all class accuracy')
legend = ax.legend(loc='upper centre', shadow=False, fontsize='x-large')
##legend.get_frame().set_facecolor('C9')
plt.grid(True)


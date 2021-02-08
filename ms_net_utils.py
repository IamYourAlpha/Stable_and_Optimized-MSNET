# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:39:14 2021

@author: intisar
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ['return_topk_args_from_heatmap', 'heatmap', 'save_checkpoint', 'calculate_matrix']

def return_topk_args_from_heatmap(matrix, n, topk):
    
    visited = np.zeros((n,n))
    tuple_list = []
    value_of_tuple = []
   #max_so_far = 0
   #s = d = 0
    
    for i in range(0, 10):
        for j in range(0, 10):
            if ( not visited[i][j] and not visited[j][i] \
                and matrix[i][j]>0):
                    tuple_list.append([i,j])
                    value_of_tuple.append(matrix[i][j])
                    visited[i][j] = 1
                    visited[j][i] = 1
                    
                    
    
    
#    for top in range(topk):
#        max_so_far = 0
#        for i in range(10):
#            for j in range(10):
#                if (matrix[i][j] > max_so_far and \
#                    visited[i][j] != 1 and visited[j][i] != 1):
#                    max_so_far = matrix[i][j]
#                    s = i
#                    d = j
#                    visited[i][j] = 1
#                    visited[j][i] = 1
    
#        tuple_list.append([s, d])
#        value_of_tuple.append(max_so_far)
    
    
    return tuple_list, value_of_tuple



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
   

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False) 
    plt.show()
    if not os.path.exists('checkpoint/figures/'):
        os.makedirs('checkpoint/figures/')
    #figure_name = 'checkpoint/figures/heatmap_%s.png'%str(depth)
    #plt.savefig(figure_name)
    return im, cbar




def calculate_matrix(model, test_loader_single, num_classes, cuda):
    model.eval()
    stop_at = 50
    tot = 0
    freqMat = np.zeros((num_classes, num_classes))
    for dta, target in test_loader_single:
        if cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output = model(dta)
        output = F.softmax(output)
        pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
        pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
        if (pred2.cpu().numpy()[0] == target.cpu().numpy()[0]):
            s = pred2.cpu().numpy()[0]
            d = pred1.cpu().numpy()[0]
            freqMat[s][d] += 1
            freqMat[d][s] += 1
            tot = tot + 1
        if (tot == stop_at):
            break
    return freqMat



def save_checkpoint(model_weights, is_best, filename='checkpoint.pth.tar'):
    filepath = os.path.join("checkpoint", filename)
    print (filepath)
    #torch.save(model_weights, filepath)
    if is_best:
        torch.save(model_weights, filepath)
        print ("******* Saved New PTH *********")
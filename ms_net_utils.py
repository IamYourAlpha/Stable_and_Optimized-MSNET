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
import pandas as pd
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2, torch
from torchvision.transforms.transforms import Resize

__all__ = ['return_topk_args_from_heatmap', 'heatmap', 'save_checkpoint', 'calculate_matrix',\
           'make_list_for_plots', 'to_csv', 'imshow']

def sort_by_value(dict_, reverse=False):
    
    dict_tuple_sorted = {k: v for k, v in \
                         sorted(dict_.items(),\
                                reverse=True, key=lambda item: item[1])}
    return dict_tuple_sorted

def return_topk_args_from_heatmap(matrix, n, topk, binary_=True):
    ''' 
    params ---- 
    matrix: interclass correlation matrix
    n: No of classes in dataset
    topk: How many expert do you want?

    return ----
    new_tuple: tuples of classes that are confusing to the router network
    value_of_tuple: Value corresponding to each tuples.
    '''

    visited = np.zeros((n,n))
    tuple_list = []
    value_of_tuple = []
    dict_tuple = {}

    for i in range(0, n):
        for j in range(0, n):
            if ( not visited[i][j] and not visited[j][i] and matrix[i][j]>0):
                dict_tuple[str(i) + "_" + str(j)] =  matrix[i][j]
                visited[i][j] = 1
                visited[j][i] = 1
                    
    dict_sorted = sort_by_value(dict_tuple, reverse=True)    
    for k, v in dict_sorted.items():
        sub_1, sub_2 = k.split("_")
        sub_1, sub_2 = int(sub_1), int(sub_2)
        tuple_list.append([sub_1, sub_2])
        value_of_tuple.append(v)
        if (len(tuple_list) == topk):
            break
    if (binary_):
        return tuple_list, value_of_tuple
    ''' if not binary we go further and 
    return more expert classes
    '''
    new_tuple = tuple_list.copy()
    for keys in tuple_list:
        key1, key2 =  keys
        temp1 = set()
        temp2 = set()
        temp1.add(key1)
        temp1.add(key2)
        temp2.add(key2)
        temp2.add(key1)
        for keys2 in tuple_list:
            if (key1 in keys2):
                first_elem, second_elem = keys2
                temp1.add(first_elem)
                temp1.add(second_elem)

        for keys2 in tuple_list:
            if (key2 in keys2):
                first_elem, second_elem = keys2
                temp1.add(first_elem)
                temp1.add(second_elem)
        if (len(temp1)>2 and (list(temp1) not in tuple_list and list(temp1) not in new_tuple)):
            new_tuple.append(list(temp1))
        if (len(temp2)>2 and (list(temp2) not in tuple_list and list(temp2) not in new_tuple)):
            new_tuple.append(list(temp2))

    return new_tuple, value_of_tuple
    #return tuple_list, value_of_tuple

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

def make_list_for_plots(lois, plot, indexes):
    lst = []
    for loi in lois:
        for index in indexes:
            ind = loi + index
            lst.append(ind)
            plot[ind] = []
    
    return plot, lst

def calculate_matrix(model, test_loader_single, num_classes, cuda, only_top2=True):
    model.eval()
    stop_at = 100
    tot = 0
    freqMat = np.zeros((num_classes, num_classes))
    for dta, target in test_loader_single:
        if cuda:
            dta, target = dta.cuda(), target.cuda()
        with torch.no_grad():
            output = model(dta)
            output = F.softmax(output, dim=1)
            pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
            pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
            if (only_top2):
                if (pred2.cpu().numpy()[0] == target.cpu().numpy()[0]):
                    s = pred2.cpu().numpy()[0]
                    d = pred1.cpu().numpy()[0]
                    freqMat[s][d] += 1
                    freqMat[d][s] += 1
            else:
                if (pred1.cpu().numpy()[0] != target.cpu().numpy()[0]):
                    s = pred1.cpu().numpy()[0]
                    d = target.cpu().numpy()[0]
                    freqMat[s][d] += 1
                    freqMat[d][s] += 1
            #     tot = tot + 1    
            # if (tot == stop_at):
            #     break
    return freqMat

def imshow(img, f_name, fexpertpred=None, fexpertconf=None, frouterpred=None, frouterconf=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() # pytorch tensor is usually (n, C(0), H(1), W(2))
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # numpy needs (H(1), W(2), C(0))
    pred = str(fexpertpred)
    conf = str(fexpertconf)
    pred_r = str(frouterpred)
    pred_c = str(frouterconf)
    res = "Experts prediction: " + pred + " : " + conf + "\n" + "Routers prediction: " + pred_r + ":" + pred_c
    #plt.text(0,0, res, color='r')
    plt.text(0, 0, res, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.savefig(f_name)
    plt.close()
    
def save_checkpoint(model_weights, is_best, filename='checkpoint.pth.tar'):
    filepath = os.path.join("checkpoint_experts", filename)
    print (filepath)
    #torch.save(model_weights, filepath)
    if is_best:
        torch.save(model_weights, filepath)
        print ("******* Saved New PTH *********")

def to_csv(plots, name):  
    data_ = pd.DataFrame.from_dict(plots)
    data_.to_csv(name, index=False)



# This part of CAM script was taken from  https://github.com/chaeyoung-lee/pytorch-CAM
# I justed merge with my project.

# generate class activation mapping for the top1 prediction

'''
   TODOS:
     Merge following function with the stable_msnet.py script
     Perform CAM analysis for image mistake by router and corrected
     by the expert network.
'''

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    print (probs, idx)
    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.5
    cv2.imwrite('F:/Research/PHD_AIZU/tiny_ai/gradcam/pytorch-CAM/cam.jpg', result)

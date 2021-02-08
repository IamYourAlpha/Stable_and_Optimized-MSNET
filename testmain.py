# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:59:16 2021

@author: intisar
"""
'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
#from __future__ import print_function

import torch
from torchvision import datasets, transforms
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
import numpy as np

state = {}


def main():
    matrix = [[1,2], [2,3]]
    _, _, _ = prepeare_dataset_for_experts(matrix)


def prepeare_dataset_for_experts(matrix):
    
    ''' there are two options , either use fixed sampler or 
    use weighted sampler. The purpose of fixed sampler is just
    to sample from specific classes, however with weighted sampler
    we can sampler from all the classes with weight. We can put more
    weight of the expert classes.
    '''
    
    data = 'cifar10'
    print('==> Preparing dataset %s' % data)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if data == 'cifar10':
        dataloader = datasets.CIFAR10
    else:
        data = datasets.CIFAR100
    
    
    train_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
    test_set = dataloader(root='./data', train=False, download=False, transform=transform_test)
    
    
    train_loader_expert = {}
    test_loader_expert = {}
    list_of_index = []
    
    class_sample_count = np.array([len(np.where(test_set.targets == t)[0]) \
                                       for t in np.unique(test_set.targets)])
        
    print ("class sample count: {}".format(class_sample_count))
    
    # Put more weights on subset class
    weight = 1. / class_sample_count
    weight[2] *= 200
    print ("Weights: {} ".format(weight))
    samples_weight = np.array([weight[t] for t in test_set.targets])
    samples_weight = torch.from_numpy(samples_weight)
    #samples_weight = samples_weight.double()
    print ("Samples weight: {} and shape: {}".format(samples_weight, len(samples_weight)))
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    
    
    
    # if subsetRandomSampler
    
    
    
    for sub in matrix: 
        indices_train = [i for i,e in enumerate(train_set.targets) if e in sub] 
        indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
        index = str(sub[0]) + "_" + str(sub[1])
        print ("The subs are {} and {}:".format(sub[0], sub[1]))
        train_loader_expert[index] = torch.utils.data.DataLoader(
                                 train_set,
                                 batch_size=16,
                                 sampler = SubsetRandomSampler(indices_train))
        test_loader_expert[index] = torch.utils.data.DataLoader(
                                 test_set,
                                 batch_size=16,
                                 sampler = SubsetRandomSampler(indices_test))
        list_of_index.append(index)
    
    # else
    # weighted sampler


    return train_loader_expert, test_loader_expert, list_of_index


if __name__ == '__main__':
    main()
    
    
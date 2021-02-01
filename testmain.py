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

state = {}


def main():
    print (state)
    key = "intisar"
    value = 29
    update_state(key, value)
    print (state)

def update_state(key, value):
    state[key] = value


if __name__ == '__main__':
    main()
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:59:11 2021

@author: intisar chowdhury
inst: The University of Aizu.
"""

# In[]:
from __future__ import print_function
import argparse

# torch
import csv
from genericpath import exists
from os import path
from os.path import basename
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#from torch.optim import optimizer
from torchvision import datasets#, transforms
import transforms
from torch.autograd import Variable
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
import os
import random
import time
import json
import copy
import numpy as np
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
# models
import models.cifar as models

# from ms_net_utils import return_topk_args_from_heatmap, heatmap, save_checkpoint, calculate_matrix,  make_list_for_plots, 
#     to_csv, 
#     imshow
from utils.ms_net_utils import *
from utils.data_utils import *



parser = argparse.ArgumentParser(description='Stable MS-NET')

# Hyper-parameters
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')


parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--train_end_to_end', action='store_true',
                    help='train from router to experts')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--router_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--corrected_images', type=str, default='./corrected_images/')
###############################################################################
parser.add_argument('--expert_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train experts')
##########################################################################
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--initialize_with_router', action='store_true', default=True)

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--evaluate_only_router', action='store_true',
                    help='evaluate router on testing set')

parser.add_argument('--weighted_sampler', action='store_true',
                    help='what sampler you want?, subsetsampler or weighted')

parser.add_argument('--finetune_experts', action='store_true', default=True,
                    help='perform fine-tuning of layer few layers of experts')
parser.add_argument('--save_images', action='store_true', default=True)
###########################################################################
parser.add_argument('--train_mode', action='store_true', default=True, help='Do you want to train or test?')

parser.add_argument('--alpha_prob', type=int, default=10, help='alpha probability')

parser.add_argument('--topk', type=int, default=10, metavar='N',
                    help='how many experts you want?')
###########################################################################

parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')


parser.add_argument('-d', '--dataset', default='cifar100', type=str)
# Architecture details
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='backbone architecture')

parser.add_argument('--depth', type=int, default=8, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                    help='initial learning rate to train')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-gpu', '--gpu_id', default=0, type=str, help='set gpu number')


#########################
    # Random Erasing
    # Turns it on if you wish to boost performance a bit like 1%
parser.add_argument('--p', default=0.3, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.3, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
state = str(state)
with open('state.txt', 'w') as f:
    f.write(state)
f.close()
model_weights = {}
use_cuda = torch.cuda.is_available()


classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# class_rev = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
#            4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


class_rev = {0: 'zero', 1: 'one', 2: 'two', 3: 'three',
           4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


if (use_cuda):
    args.cuda = True
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), 
                        F.softmax(teacher_scores/T, dim=1)) \
                        * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def get_bernouli_list(alpha_prob):
    bernouli = []
    for i in range(alpha_prob):
        bernouli.append(1) # Pr(beta) == X.X
    for i in range(alpha_prob, 100):
        bernouli.append(0) 
    return bernouli


def train(epoch, model, teacher, train_loader, train_loader_all_data, optimizer, bernouli, stocastic_loss=False): 
    model.train()
    teacher.eval()
    loss_fn = distillation
    #correct = 0
    #print ("\n \n {} Stocastic Loss is : {} {} \n \n ".format("*"*20, stocastic_loss, "*"*20))
    for batch_idx, (dta, target) in enumerate(train_loader):
        #dta_all, target_all = next(iter(train_loader_all_data))
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
            #dta_all, target_all = dta_all.cuda(), target_all.cuda()
            
        dta, target = Variable(dta), Variable(target)
        #dta_all, target_all = Variable(dta_all), Variable(target_all)
        optimizer.zero_grad()
        output = model(dta)
        #output_all = model(dta)

        if (stocastic_loss):
            #alp = random.choice(bernouli)
            output_teacher = teacher(dta) # disable knowledge dist.
            output_teacher = output_teacher.detach()
            loss = loss_fn(output, target, output_teacher, T=2, alpha=0.5)
            #loss_ce = F.cross_entropy(output, target)
            #loss = alp * loss_kd + (1-alp) * loss_ce
        else:
            loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(dta), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
    return model, optimizer


def test(model, test_loader, best_so_far, name, save_wts=False):
    model.eval()
    test_loss = 0
    correct = 0
    found_best = False
    for dta, target in test_loader:
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output = model(dta)
        output = F.softmax(output, dim=1)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        #pred1 = torch.argsort(output, dim=1, descending=True)[0:, 1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #correct += pred1.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))
    
    if (not save_wts):
        correct = correct.double()
        return None, correct
    
    now_correct = correct.double()
    
    if best_so_far < now_correct:
        print ("best correct: ", best_so_far)
        print ("now correct: ", now_correct)
        found_best = True
        wts = copy.deepcopy(model.state_dict()) # deep copy        
        name += ".pth.tar"
        save_checkpoint(wts, found_best, name)
        best_so_far = now_correct
        
    return best_so_far, now_correct 

def make_router_and_optimizer(num_classes, load_weights=False):
    print (num_classes)
    print (args.dataset)
    model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name)
    if (load_weights):
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        print ('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(args.dataset, args.arch, args.depth))
        chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(args.dataset, args.arch, args.depth))
        model.load_state_dict(chk['state_dict'])
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                      weight_decay=5e-4)
    return model, optimizer


def load_expert_networks_and_optimizers(model, lois, num_classes):
    experts = {}
    eoptimizers = {}
    #chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(args.dataset, args.arch, args.depth))
    for loi in lois:
        experts[loi] = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name)
        
        experts[loi] = experts[loi].cuda()
        
        args.initialize_with_router = False
        if (args.initialize_with_router):
            experts[loi].load_state_dict(chk['state_dict'])

        args.finetune_experts = True
        if (args.finetune_experts):
            eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer1.parameters(), 'lr': 0.00001},
                                        {'params': experts[loi].layer2.parameters(), 'lr': 0.00001},
                                         {'params': experts[loi].layer3.parameters(), 'lr': 0.1},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.1, momentum=0.9, weight_decay=5e-4)
            
        else:
            eoptimizers[loi] = optim.SGD(experts[loi].parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4)
            
        
    return experts, eoptimizers



def load_teacher_network():
    """ return the best teacher network with state_dict. """

    teacher = models.__dict__['resnext'](
                    cardinality=8,
                    num_classes=10,
                    depth=29,
                    widen_factor=4,
                    dropRate=0,
                )
    teacher = torch.nn.DataParallel(teacher).cuda()
    # checkpoint = torch.load("./ck_backup/teachers/resnext_best.pth.tar")
    # teacher.load_state_dict(checkpoint['state_dict'])
    return teacher

        
def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) #/ len(outputs)
    

def inference_with_experts_and_routers(test_loader, experts, router, topk=2):

    """ function to perform evaluation with experts
    params
    -------
    test_loader: data loader for testing dataset
    experts: dictionary of expert Neural Networks
    router: router network
    topK: upto how many top-K you want to re-check?
    """
    freqMat = np.zeros((100, 100)) # -- debug
    router.eval()
    experts_on_stack = []
    expert_count = {} 
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
        expert_count[k] = 0
    
    count = 0
    ext_ = '.png'
    correct = 0
    by_experts, by_router = 0, 0
    mistake_by_experts, mistake_by_router = 0, 0
    agree, disagree = 0, 0

    for dta, target in test_loader:
        count += 1
        # if (count == 500):
        #     break
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output_raw = router(dta)
        output = F.softmax(output_raw)
        router_confs, router_preds = torch.sort(output, dim=1, descending=True)
        preds = []
        confs = []
        for k in range(0, topk):
            #ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            ref = router_preds[0:, k]
            conf = router_confs[0:, k]
            preds.append(ref.detach().cpu().numpy()[0]) # simply put the number. not the graph
            confs.append(conf.detach().cpu().numpy()[0])
    
        cuda0 = torch.device('cuda:0')
        experts_output = []
        router_confident = True
        for exp_ in experts_on_stack:
            if (str(preds[0]) in exp_ and str(preds[1]) in exp_):
                router_confident = False
                break
        
        list_of_experts = []
        target_string = str(target.cpu().numpy()[0])
        for exp in experts_on_stack: #
            if (target_string in exp and (str(preds[0]) in exp or str(preds[1]) in exp)):
                router_confident = False
                list_of_experts.append(exp)
                expert_count[exp] += 1
                #break

        if (router_confident):
            if (preds[0] == target.cpu().numpy()[0]):
                correct += 1
                by_router += 1
            else:
                mistake_by_router += 1
                    
        else:
            for exp in experts_on_stack: #and
                if ( (str(preds[0]) in exp and str(preds[1]) in exp)):                        
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    break
           
            #and target_string in str(preds[0]) and target_string in str(preds[1])
            experts_output = [experts[exp_](dta) for exp_ in list_of_experts]
            experts_output.append(output_raw)
            experts_output_avg = average(experts_output)
            experts_output_prob = F.softmax(experts_output_avg, dim=1)
            #pred = torch.argsort(experts_output_prob, dim=1, descending=True)[0:, 0]
            exp_conf, exp_pred = torch.sort(experts_output_prob, dim=1, descending=True)
            pred, conf_ = exp_pred[0:, 0], exp_conf[0:, 0]

            if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                correct += 1
                by_experts += 1
            else:
                freqMat[pred.cpu().numpy()[0]][target.cpu().numpy()[0]]  += 1
                freqMat[target.cpu().numpy()[0]][pred.cpu().numpy()[0]]  += 1
                mistake_by_experts += 1
            if (pred.cpu().numpy()[0]  == preds[0] \
                and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                agree += 1
            elif (pred.cpu().numpy()[0]  != preds[0]\
                  and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                disagree += 1
                final_pred, final_conf =  pred.detach().cpu().numpy()[0], conf_.detach().cpu().numpy()[0]
     
                # Save misclassified samples
                args.save_images = False
                if (args.save_images):
                    data_numpy = dta[0].cpu() # transfer to the CPU.
                    f_name = '%d'%count + '%s'%ext_ # set file name with ext
                    f_name_no_text = '%d'%count + 'no_text' + '%s'%ext_
                    if (not os.path.exists(args.corrected_images)):
                        os.makedirs(args.corrected_images)
                    imshow(data_numpy, os.path.join(args.corrected_images, f_name), \
                        os.path.join(args.corrected_images, f_name_no_text), \
                        fexpertpred=class_rev[final_pred], fexpertconf=final_conf, \
                            frouterpred=class_rev[preds[0]], frouterconf=confs[0])
                    
            
    print ("Router and experts agrees with {} samplers \n and router and experts disagres for {}".format(agree, disagree))        
    print ("Routers: {} \n Experts: {}".format(by_router, by_experts))
    print ("Mistakes by Routers: {} \n Mistakes by Experts: {}".format(mistake_by_router, mistake_by_experts))
    print (expert_count)
    print (correct)
    return correct, freqMat, disagree

def ensemble_inference(test_loader, experts, router):
    router.eval()
   
    experts_on_stack = []
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
    
    correct = 0
    test_loss = 0
    list_of_experts = []
    for dta, target in test_loader:
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        
        output = router(dta)
        for exp in experts_on_stack:
            list_of_experts.append(experts[exp])
        all_outputs = [exp_(dta) for exp_ in list_of_experts]
        ##all_outputs.append(output)
        all_outputs_avg = average(all_outputs)
        all_output_prob = F.softmax(all_outputs_avg)
        output = F.softmax(output)
        test_loss += F.cross_entropy(all_output_prob, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))

def adjust_learning_rate(epoch, optimizer):
    if epoch in args.schedule:
        print ("\n\n***************CHANGED LEARNING RATE TO\n*********************\n")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        for param in optimizer.param_groups:
            print ("Lr {}".format(param['lr']))

def main():

    global best_acc
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    train_loader_router, test_loader_router, num_classes, test_loader_single = prepare_dataset_for_router(args.dataset, \
        args.train_batch, args.test_batch)
    
    print("==> creating model")
    
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=False)
    size_of_router = sum(p.numel() for p in router.parameters() if p.requires_grad == True)
    print ("Network size {:.2f}M".format(size_of_router/1000000))
    #########################################################################
    matrix = np.array(calculate_matrix(router, test_loader_single, num_classes, args.cuda, only_top2=False), dtype=int)
    #####################################################################
    print ("Calculating the heatmap for confusing class .....\n")
    ls = np.arange(num_classes)
    heatmap(matrix, ls, ls) # show heatmap
    ####################################################################################################
    matrix_args, values = return_topk_args_from_heatmap(matrix, num_classes, args.topk, binary_=True)

    print ("*"*50)
    for mat in matrix_args:
        print (mat)
    print ("*"*50)
    
    expert_train_dataloaders,  expert_test_dataloaders, lois = prepeare_dataset_for_experts(
                                                                    args.dataset,
                                                                    matrix_args, 
                                                                    values,
                                                                    args.train_batch, 
                                                                    args.test_batch, 
                                                                    weighted_sampler=True)
    print (lois)
    experts, eoptmizers = load_expert_networks_and_optimizers(router, lois, num_classes)
    teacher = load_teacher_network()
    
    args.evaluate_only_router = False
    if (args.evaluate_only_router):
        test(router, 
             test_loader_router, 
             best_so_far=None, 
             name='_', 
             save_wts=False)
        return
    
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=False)
    indexes = ['_test_experts', '_test_all']
    plot = {}
    plots, lst = make_list_for_plots(lois, plot, indexes)       
    bernouli = get_bernouli_list(args.alpha_prob)
    
    args.train_mode = True
    if (args.train_mode):
        print ("\n \n {}The value of alpha: {} \n \n ".format("*"*20, args.alpha_prob, "*"*20))
        for loi in lois:
            best_so_far = 0.0
            garbage = 99999999
            for epoch in range(1, args.expert_epochs + 1):
                adjust_learning_rate(epoch, eoptmizers[loi])
                experts[loi], eoptmizers[loi] = train(epoch,
                                                    experts[loi],
                                                    router, 
                                                    expert_train_dataloaders[loi],
                                                    train_loader_router, 
                                                    eoptmizers[loi], 
                                                    bernouli, 
                                                    stocastic_loss=False)

                best_so_far, test_acc_on_expert_data = test(experts[loi], 
                                                        expert_test_dataloaders[loi],
                                                        best_so_far, 
                                                        loi, 
                                                        save_wts=True)
                
                index = loi + '_test_experts'
                test_acc_on_expert_data_ = int(test_acc_on_expert_data.cpu().numpy())
                plots[index].append(test_acc_on_expert_data_)
                
                _, c = test(experts[loi], test_loader_router, \
                                    garbage, loi, save_wts=False)
                c_ = int(c.cpu().numpy())
                index = loi + '_test_all'
                plots[index].append(c_)
            
        ''' naming convention:
        numberOfexperts_typeofexperts_w/woKD
        '''

        #filename = 'oracle_resnet110_stocasticloss_fine_tuning_weightedsampler_cifar_100.csv'
    #filename = 'r110_svhn_random_init_subset.csv'
    #to_csv(plots, filename)
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    
    print ("*" * 50)
    best_so_far = 0
    base_location = 'checkpoint_experts'
    pth_folder = ''
    #pth_folder = 'cybercon/%s/r%s'%(args.dataset, args.depth)
    pth_exists = False
    for loi in lois:
        _, temp = test(router, expert_test_dataloaders[loi], best_so_far, "router", save_wts=False)
        print ("Performance of ROUTER in classes {} : {}".format(loi, temp))
        wts_loc = os.path.join(base_location, pth_folder, '%s'%loi + '.pth.tar')
        pth_exists = os.path.exists(wts_loc)
        if (pth_exists):
            wts = torch.load(wts_loc)
        else:
            continue
        #wts = torch.load(os.path.join(base_location, pth_folder, '%s'%loi + '.pth.tar'))
        print ("{} \n \n ".format("*" * 50))   
        experts[loi].load_state_dict(wts)
        _, temp = test(experts[loi], expert_test_dataloaders[loi], best_so_far, loi, save_wts=False)
        _, temp = test(experts[loi], test_loader_router, best_so_far, loi, save_wts=False) # delete
        _, temp = test(router, expert_test_dataloaders[loi], best_so_far, "router", save_wts=False)
        #print ("{} \n \n ".format("*" * 50))
        #print ("Performance of EXPERTS in classes {} : {}".format(loi,  temp ))
    
    #ensemble_inference(test_loader_router, experts, router)
    print ("Setting up to performance inference with experts and routers .... \n")
    topk = 2
    accuracy_exp, m, corrected_samples = inference_with_experts_and_routers(test_loader_single, experts, router, topk)
    heatmap(m, ls, ls)
    _, accuracy_router = test(router, test_loader_router, best_so_far, "router", save_wts=False)
    print ("Performance ---> Router ACC: {} \n ....  Pseudo Experts: {}\n".format(accuracy_router, accuracy_exp))
    print ("## Actual performance of experts with router: {}".format(accuracy_router + corrected_samples))


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:59:11 2021

@author: intisar
"""

# In[]:
from __future__ import print_function
import argparse

# torch
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
# models
import models.cifar as models

from ms_net_utils import return_topk_args_from_heatmap, heatmap, save_checkpoint, calculate_matrix
#from ms_net_utils import *
parser = argparse.ArgumentParser(description='Stable MS-NET')

# Hyper-parameters
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--train_end_to_end', action='store_true',
                    help='train from router to experts')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--router_epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--expert_epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train experts')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--evaluate_only_router', action='store_true',
                    help='evaluate router on testing set')

parser.add_argument('--weighted_sampler', action='store_true',
                    help='what sampler you want?, subsetsampler or weighted')

parser.add_argument('--finetune_experts', action='store_true',
                    help='perform fine-tuning of layer few layers of experts')

parser.add_argument('--topk', type=int, default=1, metavar='N',
                    help='how many experts you want?')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# Architecture details
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                    help='initial learning rate to train')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-gpu', '--gpu_id', default=0, type=str, help='set gpu number')


args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

model_weights = {}


use_cuda = torch.cuda.is_available()


classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    


if (use_cuda):
    args.cuda = True
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def doPlot(val_scores):

    scores = []
    scores = [ h.cpu().numpy() for h in val_scores]
    print (scores)
    plt.title("Teacher CNN")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, args.epochs+1), scores)
    plt.ylim( (0,1.) )
    plt.xticks(np.arange(1, args.epochs+1, 1.0))
    plt.show()


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), 
                        F.softmax(teacher_scores/T, dim=1)) \
                        * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)



def train(epoch, model, teacher, train_loader, optimizer): 
    model.train()
    #loss_fn = distillation
    correct = 0
    for batch_idx, (dta, target) in enumerate(train_loader):
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta), Variable(target)
        optimizer.zero_grad()
        output = model(dta)
       # output_teacher = teacher(dta)
       #output_teacher = output_teacher.detach()
       # loss = loss_fn(output, target, output_teacher, T=10, alpha=0.5)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
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
        output = F.softmax(output)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
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
        wts = copy.deepcopy(model.state_dict())
        
#        if (name == 'router'):
#            full_path_to_router = './checkpoint/'+'%s'%name
#            if not os.path.exists(full_path_to_router):
#                os.makedirs(full_path_to_router)
        
        name += ".pth.tar"
        save_checkpoint(wts, found_best, name)
        best_so_far = now_correct
        
    return best_so_far, now_correct 
    
def prepare_dataset_for_router():
    
    print('==> Preparing dataset %s' % args.dataset)
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
    
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    
    
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    

    testloader_single = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers)
 
    
    return trainloader, testloader, num_classes, testloader_single


def make_router_and_optimizer(num_classes, load_weights=False):
    model = models.__dict__['preresnet'](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name)
    if (load_weights):
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        chk = torch.load('./ck_backup/cifar-10/preresnet-depth-20/checkpoint/model_best.pth.tar')
        model.load_state_dict(chk['state_dict'])
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                      weight_decay=5e-4)
    return model, optimizer


def prepeare_dataset_for_experts(matrix, values):
    
    ''' there are two options , either use fixed sampler or 
    use weighted sampler. The purpose of fixed sampler is just
    to sample from specific classes, however with weighted sampler
    we can sampler from all the classes with weight. We can put more
    weight of the expert classes.
    '''
    
    
    print('==> Preparing dataset %s' % args.dataset)
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
    
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
    else:
        dataloader = datasets.CIFAR100
    
    
    train_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
    test_set = dataloader(root='./data', train=False, download=False, transform=transform_test)
    
    class_sample_count = np.array([len(np.where(train_set.targets == t)[0]) \
                                       for t in np.unique(train_set.targets)])
    
    train_loader_expert = {}
    test_loader_expert = {}
    list_of_index = []
    args.weighted_sampler = False
    if (args.weighted_sampler):
        print ("***************Preparing weighted sampler ********************************")
        for sub in matrix:
            
            weight = class_sample_count / class_sample_count
            weight[sub[0]] *= 10#alpha can be the function of counts
            weight[sub[1]] *= 10#alpha
            
            samples_weight = np.array([weight[t] for t in train_set.targets])
            samples_weight = torch.from_numpy(samples_weight)
            print ("Samples weight: {} and \n shape: {}".format(samples_weight, len(samples_weight)))
              
            
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            
            
            index = str(sub[0]) + "_" + str(sub[1])
            print ("The subs are {} and {}:".format(sub[0], sub[1]))
            
            
            train_loader_expert[index] = torch.utils.data.DataLoader(
                                     train_set,
                                     batch_size=args.train_batch,
                                     sampler = sampler)
            
            indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
            
            test_loader_expert[index] = torch.utils.data.DataLoader(
                                     test_set,
                                     batch_size=args.test_batch,
                                     sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
        
        
        return train_loader_expert, test_loader_expert, list_of_index
 
    # if subsetRandomSampler
    else:
        for sub in matrix: 
            indices_train = [i for i,e in enumerate(train_set.targets) if e in sub] 
            indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
            index = str(sub[0]) + "_" + str(sub[1])
            print ("The subs are {} and {}:".format(sub[0], sub[1]))
            train_loader_expert[index] = torch.utils.data.DataLoader(
                                     train_set,
                                     batch_size=args.train_batch,
                                     sampler = SubsetRandomSampler(indices_train))
            test_loader_expert[index] = torch.utils.data.DataLoader(
                                     test_set,
                                     batch_size=args.test_batch,
                                     sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
    
        return train_loader_expert, test_loader_expert, list_of_index


def load_expert_networks_and_optimizers(model, lois):
    experts = {}
    eoptimizers = {}
    for loi in lois:
        experts[loi] = model
        # for now
        args.finetune_experts = False
        if (args.finetune_experts):
            eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer2.parameters()},
                                         {'params': experts[loi].layer3.parameters()},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.001, momentum=0.9, weight_decay=5e-4)
            # usually lets learning rateto 0.001 for finetuning with subset sampler
            # set it to very lowe value, say 0.00001
        else:
            eoptimizers[loi] = optim.SGD(experts[loi].parameters(), lr=0.001, momentum=0.9,
                      weight_decay=5e-4)
        
    return experts, eoptimizers



def load_teacher_network():
    teacher = models.__dict__['resnext'](
                    cardinality=8,
                    num_classes=10,
                    depth=29,
                    widen_factor=4,
                    dropRate=0,
                )
    teacher = torch.nn.DataParallel(teacher).cuda()
    checkpoint = torch.load("./ck_backup/teachers/resnext_best.pth.tar")
    teacher.load_state_dict(checkpoint['state_dict'])
    return teacher



def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
     
        
def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)
    

def inference_with_experts_and_routers(test_loader, experts, router, topk):
    freqMat = np.zeros((10, 10)) # debug
    router.eval()
    experts_on_stack = []
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
    count = 0
    correct = 0
    count_of_experts = 0
    by_experts, by_router = 0, 0
    mistake_by_experts, mistake_by_router = 0, 0
    
    ''' debug mode on'''
    agree, disagree = 0, 0

    for dta, target in test_loader:
        count += 1
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output_raw = router(dta)
        output = F.softmax(output_raw)
        
        preds = []
        for k in range(0, topk):
            ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            preds.append(ref.cpu().numpy()[0])
            
        
        cuda0 = torch.device('cuda:0')
        experts_output = torch.zeros([1, 10], dtype=torch.float32, device=cuda0)
        
        router_confident = True
        for exp_ in experts_on_stack:
            if (str(preds[0]) in exp_ and str(preds[1]) in exp_):
               ## print ("Inside router block: Top 2 of router {}---{}".format(preds[0], preds[1]))
                router_confident = False
                break
        
        if (router_confident):
            if (preds[0] == target.cpu().numpy()[0]):
                correct += 1
                by_router += 1
            else:
                mistake_by_router += 1
                    
        else:
            list_of_experts = []
            for exp in experts_on_stack:
                if (str(preds[0]) in exp and str(preds[1]) in exp):
                    #experts_output_prob = F.softmax(experts[exp](dta))
                    list_of_experts.append(experts[exp])
                  #  print ("Expert block: {}---{}".format(preds[0], preds[1]))
                    count_of_experts += 1
            #print (len(list_of_experts))
            experts_output = [exp_(dta) for exp_ in list_of_experts]
            experts_output.append(output_raw)
            experts_output_avg = average(experts_output)
            experts_output_prob = F.softmax(experts_output_avg, dim=1)
            
            pred = torch.argsort(experts_output_prob, dim=1, descending=True)[0:, 0]
            if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                correct += 1
                by_experts += 1
            else:
                freqMat[pred.cpu().numpy()[0]][target.cpu().numpy()[0]]  += 1
                freqMat[target.cpu().numpy()[0]][pred.cpu().numpy()[0]]  += 1
                mistake_by_experts += 1
                print ("Predicted --> {}, actualy --> {}".format(pred.cpu().numpy()[0],\
                       target.cpu().numpy()[0]))
            
        
            if (pred.cpu().numpy()[0]  == preds[0] \
                and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                agree += 1
            elif (pred.cpu().numpy()[0]  != preds[0]\
                  and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                disagree += 1
            
            
#     
#        if (count == 1000):
#            break
    print ("Rounter and experts agrees with {} samplers \n and router and experts disagres for {}".format(agree, disagree))        
    print ("Routers: {} \n Experts: {}".format(by_router, by_experts))
    print ("Mistakes by Routers: {} \n Mistakes by Experts: {}".format(mistake_by_router, mistake_by_experts))
    
    return correct, freqMat

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
        all_outputs.append(output)
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


def make_list_for_plots(lois, plot, indexes):
    lst = []
    for loi in lois:
        for index in indexes:
            ind = loi + index
            lst.append(ind)
            plot[ind] = []
    
    return plot, lst


def to_csv(plots, name):  
    data_ = pd.DataFrame.from_dict(plots)
    data_.to_csv(name, index=False)



def main():
 
    global best_acc
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    train_loader_router, test_loader_router, num_classes, test_loader_single = prepare_dataset_for_router()
    print("==> creating model")
    
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    

    if (args.train_end_to_end):
        best_so_far = 0.0
        for epoch in range(1, args.router_epochs + 1):
            router, roptimizer = train(epoch, router, train_loader_router, roptimizer)
            best_so_far = test(router, test_loader_router, best_so_far)
    
    matrix = np.array(calculate_matrix(router, test_loader_single, num_classes, args.cuda), dtype=int)
    print ("Calculating the heatmap for confusing class .....\n")

    ls = np.arange(num_classes)
    heatmap(matrix, ls, ls) # show heatmap
    matrix_args, values = return_topk_args_from_heatmap(matrix, num_classes, args.topk)
    for mat in matrix_args:
        print (mat) 
        
    expert_train_dataloaders,  expert_test_dataloaders, lois = prepeare_dataset_for_experts(matrix_args, values)
    print(expert_train_dataloaders)
    experts, eoptmizers = load_expert_networks_and_optimizers(router, lois)
    
    
    teacher = load_teacher_network()
    ##args.evaluate_only_router = True
    if (args.evaluate_only_router):
        test(teacher, expert_test_dataloaders[lois], roptimizer, "router")
        return 
    
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    
    
    indexes=['_test_experts', '_test_all']
    plot = {}
    plots, lst = make_list_for_plots(lois, plot, indexes)       
    for loi in lois:
        best_so_far = 0.0
        garbage = 99999999
        for epoch in range(1, args.expert_epochs + 1):
            experts[loi], eoptmizers[loi] = train(epoch, \
                   experts[loi], teacher, expert_train_dataloaders[loi], eoptmizers[loi])
            
            
            best_so_far, test_acc_on_expert_data = test(experts[loi], expert_test_dataloaders[loi], \
                               best_so_far, loi, save_wts=True)
            
            index = loi + '_test_experts'
            test_acc_on_expert_data_ = int(test_acc_on_expert_data.cpu().numpy())
            plots[index].append(test_acc_on_expert_data_)
            
            _, c = test(experts[loi], test_loader_router, \
                               garbage, loi, save_wts=False)
            c_ = int(c.cpu().numpy())
            index = loi + '_test_all'
            plots[index].append(c_)
    filename = 'kd_finetuned.csv'
    to_csv(plots, filename)
    
    
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    print ("*" * 50)
    best_so_far = 0
    for loi in lois:
        _, temp = test(router, expert_test_dataloaders[loi], best_so_far, "router", save_wts=False)
        print ("Performance of ROUTER in classes {} : {}".format(loi, temp))
        wts = torch.load('checkpoint/' + '%s'%loi + '.pth.tar')
        experts[loi].load_state_dict(wts)
        _, temp = test(experts[loi], expert_test_dataloaders[loi], best_so_far, loi, save_wts=False)
        print ("Performance of EXPERTS in classes {} : {}".format(loi,  temp ))
    
    #ensemble_inference(test_loader_router, experts, router)
    print ("Setting up to performance inference with experts and routers .... \n")
    topk = 3
    accuracy_exp, m = inference_with_experts_and_routers(test_loader_single, experts, router, topk)
    heatmap(m, ls, ls)
    _, accuracy_router = test(router, test_loader_router, best_so_far, "router", save_wts=False)
    print ("Performance ---> Router ACC: {} \n ....         Experts: {}".format(accuracy_router, accuracy_exp))



if __name__ == '__main__':
    main()
    





    
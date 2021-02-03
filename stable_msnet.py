# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:59:11 2021

@author: intisar
"""

# In[]:
from __future__ import print_function
import argparse

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
import os
import random
import time
import copy
import numpy as np
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
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--train_end_to_end', action='store_true',
                    help='train from router to experts')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--router_epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--expert_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train experts')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--evaluate_only_router', action='store_true',
                    help='evaluate router on testing set')

parser.add_argument('--finetune_experts', action='store_true',
                    help='perform fine-tuning of layer few layers of experts')

parser.add_argument('--topk', type=int, default=2, metavar='N',
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




def train(epoch, model, train_loader, optimizer): 
    model.train()

    for batch_idx, (dta, target) in enumerate(train_loader):
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta), Variable(target)
        optimizer.zero_grad()
        output = model(dta)
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
        return 100. * correct.double() / len(test_loader.dataset)
    
    now_correct = correct.double()/len(test_loader)
    
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
        
    return best_so_far
    
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


def prepeare_dataset_for_experts(matrix):
    
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
    train_loader_expert = {}
    test_loader_expert = {}
    list_of_index = []
    print (matrix)
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
        args.finetune_experts = True
        if (args.finetune_experts):
            eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer2.parameters()},
                                         {'params': experts[loi].layer3.parameters()},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.001, momentum=0.9, weight_decay=5e-4)
        else:
            eoptimizers[loi] = optim.SGD(experts[loi].parameters(), lr=0.001, momentum=0.9,
                      weight_decay=5e-4)
        
    return experts, eoptimizers



def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
            
def inference_with_experts_and_routers(test_loader, experts, router, topk):
    
    router.eval()
    experts_on_stack = []
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
    
    correct = 0
    for dta, target in test_loader:
       
        if args.cuda:
            dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output = router(dta)
        output = F.softmax(output)
        
        preds = []
        for k in range(0, topk):
            ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            preds.append(ref.cpu().numpy()[0])
            
        
        cuda0 = torch.device('cuda:0')
        experts_output = torch.zeros([1, 10], dtype=torch.float32, device=cuda0)
       
        if (str(preds[0]) not in experts_on_stack):
            if (preds[0] == target.cpu().numpy()[0]):
                correct += 1
        else:
            for exp in experts_on_stack:
                if (str(preds[0]) in exp):
                    experts_output += experts[exp](dta)
            experts_output += output
            experts_output_sm = F.softmax(experts_output)
            pred = torch.argsort(experts_output_sm, dim=1, descending=True)[0:, 0]
            if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                correct += 1

    return correct

def main():
    
    global best_acc
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    train_loader_router, test_loader_router, num_classes, test_loader_single = prepare_dataset_for_router()
    print("==> creating model")
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    
    if (args.evaluate_only_router):
        test(router, test_loader_router, roptimizer, "router")
        return 
    
    if (args.train_end_to_end):
        best_so_far = 0.0
        for epoch in range(1, args.router_epochs + 1):
            router, roptimizer = train(epoch, router, train_loader_router, roptimizer)
            best_so_far = test(router, test_loader_router, best_so_far)
    
    matrix = np.array(calculate_matrix(router, test_loader_single, num_classes, args.cuda), dtype=int)
    print ("Calculating the heatmap for confusing class .....\n")
        
    ls = np.arange(num_classes)
    heatmap(matrix, ls, ls) # show heatmap
    matrix_args = return_topk_args_from_heatmap(matrix, num_classes, args.topk)
    for mat in matrix_args:
        print (mat) 
        
    expert_train_dataloaders,  expert_test_dataloaders, lois = prepeare_dataset_for_experts(matrix_args)
    experts, eoptmizers = load_expert_networks_and_optimizers(router, lois)
    
    #training of the expert networks begin
    router, roptimizer = make_router_and_optimizer(num_classes, load_weights=True)
    save_wts = True
    for loi in lois:
        best_so_far = 0.0
        for epoch in range(1, args.expert_epochs + 1):
#            if (args.codistillation):
#                
            experts[loi], eoptmizers[loi] = train(epoch, experts[loi], expert_train_dataloaders[loi], eoptmizers[loi])
            best_so_far = test(experts[loi], expert_test_dataloaders[loi], best_so_far, loi, save_wts)
    
    
    save_wts = False
#    wts_router = torch.load('checkpoint/model_best.pth.tar')
#    router.load_state_dict(wts_router)
    for loi in lois:
        #temp = test(router, expert_test_dataloaders[loi], best_so_far, "router", save_wts)
        #print ("Performance of ROUTER in classes {} : {}".format(loi, temp))
        wts = torch.load('checkpoint/' + '%s'%loi + '.pth.tar')
        experts[loi].load_state_dict(wts)
        #temp = test(experts[loi], expert_test_dataloaders[loi], best_so_far, loi, save_wts)
        #print ("Performance of EXPERTS in classes {} : {}".format(loi,  temp ))
    
    print ("Setting up to performance inference with experts and routers .... \n")
    topk = 2
    accuracy_exp = inference_with_experts_and_routers(test_loader_single, experts, router, topk)
    accuracy_router = test(router, test_loader_router, best_so_far, "router", save_wts)
    print ("Performance ---> Router ACC: {} \n .... Experts: {}".format(accuracy_router, accuracy_exp))
#    model_size = sum( p.numel() for p in router.parameters() if p.requires_grad)
#    print("The size of the Teacher Model: {}".format(model_size))
   
if __name__ == '__main__':
    main()
    





    
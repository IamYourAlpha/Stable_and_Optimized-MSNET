# In[]:
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import torchvision
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  SubsetRandomSampler
import models.cifar as models # teacher
from utils import   accuracy
# In[]:
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#torch.manual_seed(args.seed)
#if args.cuda:
#    torch.cuda.manual_seed(args.seed)

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


classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    

    
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       # transforms.RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, ),
    ])
    
    
transform_test = transforms.Compose([
             transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    

#
#torch.manual_seed(args.seed)
#
SUB = {0, 8}

#################################################


test_set = datasets.CIFAR10('./data', train=False, download=True,
                   transform=transform_test)



train_set = datasets.CIFAR10('./data', train=True, download=True,
                   transform=transform_train)


## SELECT THE INDICES OF THE CLASS YOU WANT TO TRAIN THE EXPERT MODULE ON
indices_train = [i for i,e in enumerate(train_set.targets) if e in SUB] 
tot_train = len(indices_train)
print (tot_train)
indices_test = [j for j,k in enumerate(test_set.targets) if k in SUB]
tot_test = len(indices_test)
print (tot_test)

print (SUB)
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
                         batch_size=args.batch_size, shuffle=True)#sampler=SubsetRandomSampler(indices_train))

test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, download=True,
                         transform=transform_test),
                         batch_size=args.batch_size, sampler=SubsetRandomSampler(indices_test))

#test_loader = torch.utils.data.DataLoader(
#        datasets.CIFAR10('../data', train=False, download=True,
#                         transform=transform_test),
#                         batch_size=args.batch_size, shuffle=True)
#

  


best_correct = -999
freqMat = np.zeros((10,10))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0 
    now_correct =  0
    global best_correct
    

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output = F.softmax(output)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        pred1 = torch.argsort(output, dim=1, descending=True)[0:, 1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
       # correct += pred1.eq(target.data.view_as(pred)).cpu().sum() 
    test_loss /= len(test_loader.dataset)
    print('\nTest set top-1 : Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))

    

teacher_model = models.__dict__['resnext'](
                    cardinality=8,
                    num_classes=10,
                    depth=29,
                    widen_factor=4,
                    dropRate=0,
                )
teacher_model = torch.nn.DataParallel(teacher_model).cuda()
ck = torch.load('./ck_backup/teachers/resnext_best.pth.tar')
teacher_model.load_state_dict(ck['state_dict'])


model = models.__dict__["preresnet"](
                    num_classes=10,
                    depth=110,
                    block_name='BasicBlock',
                )
model = torch.nn.DataParallel(model).cuda()







chk = torch.load('./ck_backup/cifar-10/preresnet-depth-110/checkpoint/model_best.pth.tar')
model.load_state_dict(chk['state_dict'])

args.epochs = 1
val_scores = []
for epoch in range(1, args.epochs + 1):
     start_time = time.time()
     #print ("Resnext model: {}".format(test(teacher_model)))
     #print ("preresnet: {}".format(test(model)))
     test(model)
     test(teacher_model)




    
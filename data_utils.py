import torch
from torchvision import datasets#, transforms
import torch.utils.data as data
import numpy as np
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
import transforms

def prepare_dataset_for_router(dataset_, train_batch, test_batch):
    
    print('==> Preparing dataset %s' % dataset_)
    
    
    if dataset_ == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    
    if dataset_ == 'fmnist':
        dataloader = datasets.FashionMNIST
        num_classes = 10
    
    if dataset_ == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10

    if dataset_ == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100

    if (dataset_ == 'fmnist'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
    else:
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

    if (dataset_ == 'svhn'):
        trainset = dataloader(root='./data', split='train', download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0)
       
        testset = dataloader(root='./data', split='test', download=True, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=0)
       
        testloader_single = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
   
    else:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=False, num_workers=0)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=0)
        
        testloader_single = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
 
    if (dataset_  != 'cifar100'):
        num_classes = 10
    return trainloader, testloader, num_classes, testloader_single


def prepeare_dataset_for_experts(dataset_, matrix, values, train_batch, test_batch, weighted_sampler=True):
    
    ''' note: there are two options , either use fixed sampler or 
    use weighted sampler. The purpose of fixed sampler is just
    to sample from specific classes, however with weighted sampler
    we can sampler from all the classes with weight. We can put more
    weight of the expert classes.
    params ----
    matrix: interclass correlation/confusing class matrix
    values: valus correpdonding to the matrix

    return ----
    
    '''
    
    
    print('==> Preparing dataset %s' % dataset_)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset_ == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    
    if dataset_ == 'fmnist':
        dataloader = datasets.FashionMNIST
        num_classes = 10
    
    if dataset_ == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10

    if dataset_ == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    
    if (dataset_ == 'fmnist'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
    
    if (dataset_ == 'svhn'):
        train_set = dataloader(root='./data', split='train', download=True, transform=transform_train)
        test_set = dataloader(root='./data', split='test', download=True, transform=transform_test)

    else:
        train_set = dataloader(root='./data', train=True, download=True, transform=transform_train)
        test_set = dataloader(root='./data', train=False, download=False, transform=transform_test)
    
    if (dataset_ == 'svhn'):
        class_sample_count = np.array([len(np.where(train_set.labels == t)[0]) \
                                        for t in np.unique(train_set.labels)])
    else:
        class_sample_count = np.array([len(np.where(train_set.targets == t)[0]) \
                                        for t in np.unique(train_set.targets)])


    train_loader_expert = {}
    test_loader_expert = {}
    list_of_index = []
 
    if (weighted_sampler):
        print ("***************Preparing weighted sampler ********************************")
        for sub in matrix:
            weight = class_sample_count / class_sample_count
            for sb in sub:
                weight[sb] *= 10
            samples_weight = np.array([weight[t] for t in train_set.targets])
            samples_weight = torch.from_numpy(samples_weight)
            print ("Samples weight: {} and \n shape: {}".format(samples_weight, len(samples_weight)))
            sampler_ = WeightedRandomSampler(samples_weight, len(samples_weight))

            index = ""
            for i, sb in enumerate(sub):
                index += str(sb)
                if (i < len(sub)-1):
                    index += "_"
            print ("The subs are: {}".format(index))
            
            
            train_loader_expert[index] = torch.utils.data.DataLoader(
                                     train_set,
                                     batch_size=train_batch,
                                     sampler = sampler_)
            
            indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
            
            test_loader_expert[index] = torch.utils.data.DataLoader(
                                     test_set,
                                     batch_size=test_batch,
                                     sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
        
        return train_loader_expert, test_loader_expert, list_of_index
 
    # if subsetRandomSampler
    else:
        for sub in matrix:
            if (dataset_ == 'svhn'):
                indices_train = [i for i,e in enumerate(train_set.labels) if e in sub] 
                indices_test = [j for j,k in enumerate(test_set.labels) if k in sub]
            else:
                indices_train = [i for i,e in enumerate(train_set.targets) if e in sub] 
                indices_test = [j for j,k in enumerate(test_set.targets) if k in sub]
            index = ""
            for i, sb in enumerate(sub):
                index += str(sb)
                if (i < len(sub)-1):
                    index += "_"
            print ("The subs are in subset sampler: {}".format(index))
            
            train_loader_expert[index] = torch.utils.data.DataLoader(
                                     train_set,
                                     batch_size=train_batch,
                                     sampler = SubsetRandomSampler(indices_train))
            test_loader_expert[index] = torch.utils.data.DataLoader(
                                     test_set,
                                     batch_size=test_batch,
                                     sampler = SubsetRandomSampler(indices_test))
            list_of_index.append(index)
    
        return train_loader_expert, test_loader_expert, list_of_index
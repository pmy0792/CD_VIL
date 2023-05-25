# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

from asyncio.proactor_events import _ProactorDuplexPipeTransport
from posixpath import split
import random

import torch
from torch.utils.data.dataset import Subset
import torch.utils.data as data
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *
from torch.utils.data import ConcatDataset
from continual_datasets.dataset_utils import download_file_from_google_drive
import utils
import re

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes


def determine_task_type(prev_d,prev_c,cur_d,cur_c):
    if prev_d != cur_d and prev_c!= cur_c:
        task_type="CDIL"
    elif prev_d != cur_d:
        task_type="DIL"
    elif prev_c!= cur_c:
        task_type="CIL"
    else:
        print("unidentified config: {prev_d}, {cur_d}, {prev_c}, {cur_c}")
    return task_type

def extract_task_order(filename):
    # Open the file
    # open the text file
    with open(filename, 'r') as file:
        # read the contents of the file
        contents = file.read()
        # find all the lines with integers separated by commas
        lines = re.findall(r'\[\d+(?:,\s*\d+)*\]\s+\w+', contents)
        # create a list of integers for each line
        classes = [[int(x) for x in re.findall(r'\d+', line)] for line in lines]
        
        
        domains = []
        pattern = r"D(\d+)"

        for match in re.findall(pattern, contents):
            domains.append(int(match))
        
        
        # Print the extracted values
        print(domains)
        print(classes)
        return domains, classes

def build_continual_dataloader(args):
    
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    
        
    elif args.dataset in ['CORe50', 'DomainNet','PermutedMNIST','iDigits']: 
        dataset_train, dataset_val = get_dataset(args.dataset, transform_train, transform_val, args)
        
        #args.nb_classes = len(dataset_train[0].classes)
        
        configs = {
                    "CORe50":{"nb_classes":50, "domain":8, "class_group_size": 10, "class_group":5},
                   "DomainNet":{"nb_classes":345, "domain":6, "class_group_size": 69, "class_group":5},
                   "PermutedMNIST":{"nb_classes":10, "domain":5, "class_group_size": 2, "class_group":5},
                   "iDigits":{"nb_classes":10, "domain":4, "class_group_size": 2, "class_group":5}
                   }
        args.nb_classes = configs[args.dataset]["nb_classes"]
        args.class_group_size = configs[args.dataset]["class_group_size"]
        splited_dataset = list()
        class_mask = list()
        if args.learning_type == "random":
            for i in range(len(dataset_train)): #? len(dataset_train)== # of domains?
                dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                splited_dataset.append(dataset)
                class_mask.append(mask)
            
            splited_dataset = sum(splited_dataset, [])
            class_mask = sum(class_mask, [])

            #!
            args.num_tasks = len(splited_dataset)
            if args.shuffle:
                zipped = list(zip(splited_dataset, class_mask))
                random.shuffle(zipped)
                splited_dataset, class_mask = zip(*zipped)
                
        elif args.learning_type == "VIL":
            '''
            for i in range(len(dataset_train)): #? len(dataset_train)== # of domains?
                if args.dataset in ["PermutedMNIST","iDigits"]:
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val[i], args)
                else:
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                splited_dataset.append(dataset)
                class_mask.append(mask)
            
            splited_dataset = sum(splited_dataset, [])
            class_mask = sum(class_mask, [])

            #!
            args.num_tasks = len(splited_dataset)
            if args.shuffle:
                zipped = list(zip(splited_dataset, class_mask))
                random.shuffle(zipped)
                splited_dataset, class_mask = zip(*zipped)
                
            splited_dataset=splited_dataset[:args.num_tasks]
            class_mask=class_mask[:args.num_tasks]
            '''
            domain_order, class_group_order = extract_task_order(f"task_order/{args.dataset}/{args.seed}.txt")#"task_order/"+ args.dataset+".txt")
            for i in range(len(dataset_train)): #? len(dataset_train)== # of domains?
                if args.dataset in ["PermutedMNIST","iDigits"]:
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val[i], args)
                else:
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                splited_dataset.append(dataset)
                class_mask.append(mask)
            print(f"splited_dataset:{splited_dataset}")
            ordered_splited_dataset=[]
            ordered_class_mask=[]
            for idx,d in enumerate(domain_order): # iterate task num
                class_group_idx = int(class_group_order[idx][0]/args.class_group_size)
                ordered_splited_dataset.append(splited_dataset[d][class_group_idx])
                ordered_class_mask.append(class_mask[d][class_group_idx])
            args.num_tasks = len(ordered_splited_dataset)
            splited_dataset = ordered_splited_dataset
            class_mask=ordered_class_mask
            
            
        elif args.learning_type == "CIL_only":
            
            if args.dataset in ["PermutedMNIST","iDigits"]:
                splited_dataset, class_mask = split_single_dataset(dataset_train[0], dataset_val[0], args) 
                for i in range(1,len(dataset_train)): #? len(dataset_train)== # of domains
                    splited_single_dataset, mask = split_single_dataset(dataset_train[i], dataset_val[i], args)
                    
                    for t, train_and_val in enumerate(splited_single_dataset):
                        appended_train = ConcatDataset([splited_dataset[t][0] , splited_single_dataset[t][0]])
                        appended_val = ConcatDataset([splited_dataset[t][1] , splited_single_dataset[t][1]])
                        splited_dataset[t] =[appended_train, appended_val]
            else:
                splited_dataset, class_mask = split_single_dataset(dataset_train[0], dataset_val, args) 
                for i in range(1,len(dataset_train)): #? len(dataset_train)== # of domains
                    splited_single_dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                    
                    for t, train_and_val in enumerate(splited_single_dataset):
                        appended_train = ConcatDataset([splited_dataset[t][0] , splited_single_dataset[t][0]])
                        appended_val =  splited_single_dataset[t][1]
                        splited_dataset[t] =[appended_train, appended_val]
            
            
            if args.num_tasks < len(splited_dataset):
                splited_dataset = splited_dataset[:args.num_tasks]
                class_mask = class_mask[:args.num_tasks]
            if args.shuffle:
                zipped = list(zip(splited_dataset, class_mask))
                random.shuffle(zipped)
                splited_dataset, class_mask = zip(*zipped)
                
        elif args.learning_type == "DIL_only":
            splited_dataset=[]
            
            splited_dataset_temp=[]
            
            args.num_tasks=len(dataset_train)
            if args.dataset =="CORe50":
                for i in range(len(dataset_train)):
                    splited_dataset.append([dataset_train[i], dataset_val])
            elif args.dataset=="DomainNet":
                splited_dataset=[[] for i in range(configs[args.dataset]["class_group"])]
                for i in range(len(dataset_train)): # iterate domain
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                    splited_dataset_temp.append(dataset)
                
                splited_dataset_d=[]   
                for d in range(configs[args.dataset]["domain"]):
                    for c in range(configs[args.dataset]["class_group"]): 
                        splited_dataset_d.append(splited_dataset_temp[d][c])
                    splited_dataset.append(sum(splited_dataset_d,[]))
                    splited_dataset_d=[]
                
                print(splited_dataset)
            else:
                for i in range(len(dataset_train)):
                    splited_dataset.append([dataset_train[i], dataset_val[i]])
                    
            single_mask = [ i for i in range(args.num_classes)]
            class_mask = [single_mask for _ in range(args.num_tasks) ]
        elif args.learning_type == "CDIL_only":
            splited_dataset=[]
            class_mask=[]     
            visited=[] 
            if args.dataset in ["iDigits","PermutedMNIST"]: # iDigits, PMNIST
                splited_dataset=[]
                class_mask=[]     
                visited=[] # visited class group
                
                for i in range(len(dataset_train)): # iterate domain
                    splited_datasets,mask = split_single_dataset(dataset_train[i],dataset_val[i],args)
                    n = random.randrange(0,args.num_tasks)
                    
                    if n in visited:
                        while n in visited:
                            n = random.randrange(0,args.num_tasks)
                    splited_dataset.append(splited_datasets[n])
                    class_mask.append(mask[n])
                    visited.append(n)
            else:
                splited_dataset_temp=[]
                class_mask_temp=[]   
                for i in range(len(dataset_train)):
                    dataset, mask = split_single_dataset(dataset_train[i], dataset_val, args)
                    splited_dataset_temp.append(dataset)
                    class_mask_temp.append(mask)
                for i in range(configs[args.dataset]["class_group"]):
                    n = random.randrange(0,args.num_tasks)
                    if n in visited:
                        while n in visited:
                            n = random.randrange(0,args.num_tasks)
                    splited_dataset.append(splited_dataset_temp[n][i]) #n번째 domain, i번째 classgroup
                    class_mask.append(class_mask_temp[n][i])
                    visited.append(n)
                    
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        elif args.dataset == 'iDigits':
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
        else:
            dataset_list = args.dataset.split(',')
        
        # if args.shuffle:
        #     random.shuffle(dataset_list)
            
        print(dataset_list)
    
        args.nb_classes = 0
    
    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-') or args.dataset in ['CORe50', 'DomainNet','PermutedMNIST','iDigits']:
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                    class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])
                    args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                    dataset_train.target_transform = transform_target
                    dataset_val.target_transform = transform_target
            
        if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

        data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        '''    
        if args.learning_type=='CDIL_only':
            task_config=[]
            task_id=0
            while task_id < len(dataloader):
                data=dataloader[task_id]['train']
                device = torch.device(args.device)
                for input, target, domain,class_group in data:
                    domain = domain.to(device, non_blocking=True)[0].item() # torch CUDA-> integer
                    class_group = class_group.to(device, non_blocking=True)[0].item()
                    
                    if task_id>0: # Define task type
                        prev_domain, prev_class_group,_ = task_config[task_id-1]
                        task_type = determine_task_type(prev_domain, prev_class_group, domain, class_group)
                        
                    else:
                        task_type="Initial"
                    #print(f"task_type:{task_type}")
                    if task_type =="DIL" or task_type== "CIL":
                        del dataloader[task_id]
                        args.num_tasks-=1
                    else: #Initial of CDIL 
                        task_config.append((domain, class_group,task_type))
                        task_id+=1
                    break # check only first image data for task configuration
            
            if len(task_config)>=args.CDIL_only_num_task:
                task_config=task_config[:args.CDIL_only_num_task]
                dataloader=dataloader[:args.CDIL_only_num_task]
                args.num_tasks = args.CDIL_only_num_task
            #print(f"Task configuration for CDIL_only :\n {task_config}")
        '''
    return dataloader, class_mask, args

def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train,args=args).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val,args=args).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train,args=args).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val,args=args).data

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)
    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'PermutedMNIST':
        dataset_train = [PermutedMNIST(args.data_path, train=True, download=True, transform=transform_train, random_seed=i) for i in range(5)]
        dataset_val = [PermutedMNIST(args.data_path, train=False, download=True, transform=transform_val, random_seed=i) for i in range(5)]
        
    elif dataset == 'iDigits':
        dataset_train = [MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train),
                         SVHN(args.data_path, split='train', download=True, transform=transform_train),
                         MNISTM(args.data_path, train=True, download=True, transform=transform_train),
                         SynDigit(args.data_path, train=True, download=True, transform=transform_train)]
        dataset_val =   [MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train),
                         SVHN(args.data_path, split='test', download=True, transform=transform_train),
                         MNISTM(args.data_path, train=False, download=True, transform=transform_train),
                         SynDigit(args.data_path, train=False, download=True, transform=transform_train)]
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args): #!
    if True: # original
        nb_classes = args.nb_classes #len(dataset_val.classes)
        assert nb_classes % args.class_group_size ==0
        split_num =int(nb_classes / args.class_group_size)
        classes_per_task = args.class_group_size

        labels = [i for i in range(nb_classes)]
        
        split_datasets = list()
        mask = list()

        # if args.shuffle:
        #     random.shuffle(labels)

        for _ in range(split_num):
            train_split_indices = []
            test_split_indices = []
            
            scope = labels[:classes_per_task] #! class group
            labels = labels[classes_per_task:]
            
            #print(f"scope: {scope}")

            mask.append(scope)

            for k in range(len(dataset_train.targets)):
                if int(dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)
            
            
            for h in range(len(dataset_val.targets)):
                if int(dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)
            
            subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

            split_datasets.append([subset_train, subset_val])
    #print(f"splited_datasets:{split_datasets}\nmask:{mask}")
    return split_datasets, mask

class Permutation(object):
    """
    Defines a fixed permutation for a numpy array.
    """
    def __init__(self, randomseed) -> None:
        """
        Initializes the permutation.
        """
        self.randomseed = randomseed

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """
        Randomly defines the permutation and applies the transformation.
        :param sample: image to be permuted
        :return: permuted image
        """
        old_shape = sample.shape
        if self.randomseed is not None:
            np.random.seed(self.randomseed)
            self.perm = np.random.permutation(len(sample.flatten()))

        return sample.flatten()[self.perm].reshape(old_shape)
def build_transform(is_train, args, randomseed=None):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        t=[]
        t.append(transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio))
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.ToTensor())
        if args.dataset == 'PermutedMNIST':
            t.append(Permutation(randomseed))
        if args.dataset == "CORe50":
            t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(t)
        
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    if args.dataset == 'PermutedMNIST':
        t.append(Permutation(randomseed))
    if args.dataset == "CORe50":
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(t)
# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

#from continual_datasets.continual_datasets import *
from continual_datasets.continual_datasets_ import *
import utils

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
        args.nb_classes = len(dataset_val.classes)

    elif args.dataset in ['CORe50', 'DomainNet', 'PermutedMNIST'] and (args.domain_inc or args.versatile_inc):
        dataset_train, dataset_val = get_dataset(args.dataset, transform_train, transform_val, args)

        if args.dataset in ['CORe50']:
            splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
            args.nb_classes = len(dataset_val.classes)
        else:
            splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_train))]
            args.nb_classes = len(dataset_train[0].classes)

    # Case: Configure tasks with multiple datasets
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['CIFAR10', 'MNIST', 'FashionMNIST', 'SVHN', 'NotMNIST']

        elif args.dataset == 'iDigits':
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']

        else:
            dataset_list = args.dataset.split(',')

        if args.shuffle and not args.versatile_inc:
            random.shuffle(dataset_list)
        print(dataset_list)

        splited_dataset = list()
        args.nb_classes = 0
        for i in range(len(dataset_list)):
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)
            if not args.domain_inc and not args.versatile_inc:
                transform_target = Lambda(target_transform, args.nb_classes)
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
            splited_dataset.append((dataset_train, dataset_val))
            args.nb_classes += len(dataset_val.classes)

    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)
        for c, d in zip(class_mask, domain_list):
            print(c, d)
        
    for i in range(args.num_tasks):
        dataset_train, dataset_val = splited_dataset[i]
        
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

    return dataloader, class_mask

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
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val).data
    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'PermutedMNIST':
        transform_train = list()
        transform_val = list()
        for i in range(args.num_tasks):
            transform_train.append(build_transform(True, args, i+args.seed))
            transform_val.append(build_transform(False, args, i+args.seed))
        dataset_train = [PermutedMNIST(args.data_path, train=True, download=True, transform=transform_train[i]) for i in range(args.num_tasks)]
        dataset_val = [PermutedMNIST(args.data_path, train=False, download=True, transform=transform_val[i]) for i in range(args.num_tasks)]

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    print(nb_classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    for _ in range(args.num_tasks):
        train_split_indices = list()
        test_split_indices = list()
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def build_vil_scenario(splited_dataset, args):
    datasets = list()
    class_mask = list()
    domain_list = list()

    for i in range(len(splited_dataset)):
        dataset, mask = split_single_dataset(splited_dataset[i][0], splited_dataset[i][1], args)
        datasets.append(dataset)
        class_mask.append(mask)
        for _ in range(len(dataset)):
            domain_list.append(f'D{i}')

    splited_dataset = sum(datasets, [])
    class_mask = sum(class_mask, [])

    args.num_tasks = len(splited_dataset)

    zipped = list(zip(splited_dataset, class_mask, domain_list))
    random.shuffle(zipped)
    splited_dataset, class_mask, domain_list = zip(*zipped)
    return splited_dataset, class_mask, domain_list, args

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
    if args.dataset=="CORe50":
            t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)
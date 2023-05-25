# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import random
import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils

def mahalanobis(u, v, VI):
                    u = u.view(1, -1)  # convert u to a row vector
                    v = v.view(1, -1)  # convert v to a row vector
                    VI = torch.as_tensor(VI, dtype=torch.float32)
                    delta = u - v
                    m = torch.matmul(torch.matmul(delta, VI), delta.t())
                    return torch.sqrt(m)

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    #dataset = DomainNet(args.data_path, train=True, download=False).data #!!!!
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        #domain = domain.to(device, non_blocking=True)
        #class_group = class_group.to(device, non_blocking=True)
        
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        
        if args.d_prompt and args.two_stage=="e_then_d":
            # first forward to train only G&E prompt       
            model.module.d_prompt.prompt_key.requires_grad=False
            model.module.d_prompt.prompt_pool.requires_grad=False
            
            output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode,args=args,train_only_GE=True)
            logits = output['logits']
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            # logits: (bs, class_num), targets: (bs)
            loss = criterion(logits, target) # base criterion (CrossEntropyLoss
            if 'e_reduce_sim' in output:
                loss -= args.pull_constraint_coeff * output['e_reduce_sim']
            optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            torch.cuda.synchronize()
            
            output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode,args=args)
            logits = output['logits']
        
            if args.d_prompt:
                d_cls_features = output['d_cls_features'] 
                
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
                       
        # if args.d_prompt and not args.d_prompt_as_dualprompt:
        #     model.module.d_prompt.prompt_key.requires_grad=True
        #     model.module.d_prompt.prompt_pool.requires_grad=True
        # elif args.d_prompt and args.d_prompt_as_dualprompt:
        #     model.module.de_prompt.prompt_key.requires_grad=True
        #     model.module.g_prompt.prompt_key.requires_grad=True
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode,args=args)
        logits = output['logits']
        
        if args.d_prompt:
            d_cls_features = output['d_cls_features'] 
            
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        # logits: (bs, class_num), targets: (bs)
        loss = criterion(logits, target) # base criterion (CrossEntropyLoss) 
        if args.pull_constraint:
            if 'e_reduce_sim' in output:
                    loss -= args.pull_constraint_coeff * output['e_reduce_sim']
            if 'de_reduce_sim' in output:
                    loss -= args.pull_constraint_coeff * output['de_reduce_sim']
            if 'd_reduce_sim' in output:
                    loss -= args.pull_constraint_coeff * output['d_reduce_sim']
                
                
        if args.d_prompt:# and not args.d_prompt_as_dualprompt:
            #loss += args.alpha * torch.tensor(np.mean(pdist(d_cls_features.cpu().detach().numpy(), 'euclidean'))).to(device)
            if args.d_loss=="std":
                bs = d_cls_features.shape[0]
                centroid =  torch.mean(d_cls_features, dim=0, keepdim=True).repeat(bs,1)
                std_dev = torch.mean(torch.square(d_cls_features - centroid))
                
                loss += args.alpha * std_dev
            elif args.d_loss=="mahalanobis":
                bs, emb = d_cls_features.shape[0], d_cls_features.shape[1]
                covariance = torch.cov(d_cls_features.T)
                constant = 1e-6
                diag_constant=torch.diag_embed(torch.full((emb,), constant)).to(covariance.device)
                covariance = torch.add(covariance, diag_constant)
                inverse_covariance = torch.pinverse(covariance)
                centroid = torch.mean(d_cls_features, dim=0)
                centered_features = d_cls_features - centroid.unsqueeze(0)  # Center the features
                distance=0
                for f in centered_features:
                    distance+=mahalanobis(f,centroid,inverse_covariance)
                    
                # #!
                # distances = torch.sqrt(torch.sum(torch.matmul(centered_features, inverse_covariance) * centered_features, dim=1))
                # sum_distances = torch.sum(distances, dim=0)  
                #print(f"mahalanobisdistance:{distance.item()/bs}")
                loss =loss+ 0.0001 * (distance/bs)
            if args.de_loss=="cos_sim":
                    d_cls_features,ge_cls_features_list=output['d_cls_features'], output['GE_cls_features_list']
                    bs,emb=d_cls_features.shape
                    cos_sim=0
                    cosine_distance=0
                    
                    for idx, f in enumerate(d_cls_features): # 하나의 d-feature가 모든  ge feature와 멀어짐
                        for ge_cls_features in ge_cls_features_list:
                            cos_sim = torch.matmul(model.module.l2_normalize(d_cls_features[idx],dim=0), model.module.l2_normalize(ge_cls_features[idx],dim=0).T)
                            cosine_distance += 1-cos_sim
                    #print(cosine_distance.item()/bs)    
                    loss = loss + 2*(1/(cosine_distance/bs))
            elif args.de_loss=="l2":
                d_cls_features,ge_cls_features=output['d_cls_features'], output['GE_cls_features']
                distance = torch.norm( d_cls_features- ge_cls_features, p='fro')
                #print(distance)
                loss = loss +100*(1/distance)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    correct_d_count = 0
    batch = 0
    _targets=[]
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            #domain = domain.to(device, non_blocking=True)
            #class_group = class_group.to(device, non_blocking=True)
            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
    
            output = model(input, task_id=task_id, cls_features=cls_features, upper = False, train = False, correct_d=args.correct_d, args=args)
            
            logits = output['logits']
            #pre_logits = output['pre_logits'] # (batch, embedding)
            
            _targets=_targets+target.tolist()

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            # if args.d_prompt:
            #     #! Evaluate D-prompt selection
            #     correct_d_count += output['test_d_prompt_selection'][task_id] 
            #     batch += input.shape[0]
     
     
      
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, (correct_d_count, batch)


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    if args.eval_only_last:
        if args.num_tasks <= task_id+1 :  # last task
            for i in range(task_id+1):
                test_stats, d_selection= evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                                            device=device, task_id=i, class_mask=class_mask, args=args)
                        
                stat_matrix[0, i] = test_stats['Acc@1']
                stat_matrix[1, i] = test_stats['Acc@5']
                stat_matrix[2, i] = test_stats['Loss']

                acc_matrix[i, task_id] = test_stats['Acc@1']
        else: 
            test_stats= dict()
    
    else : 
        for i in range(task_id+1):
                test_stats, d_selection = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                                            device=device, task_id=i, class_mask=class_mask, args=args)
                        
                stat_matrix[0, i] = test_stats['Acc@1']
                stat_matrix[1, i] = test_stats['Acc@5']
                stat_matrix[2, i] = test_stats['Loss']

                acc_matrix[i, task_id] = test_stats['Acc@1']
                
                
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    
    return test_stats


def task_configuration(data_loader: Iterable, device: torch.device, args =None,):
    
    # if args.distributed and utils.get_world_size() > 1:
    #     data_loader.sampler.set_epoch(0)
    print("---------------------------------------------Train Task Configuration---------------------------------------------")
    task_config=[]
    cil,dil,cdil=0,0,0
    for task_id in range(args.num_tasks):
        data=data_loader[task_id]['train']
        if args.distributed and utils.get_world_size() > 1:
            data.sampler.set_epoch(0)
        print(f"Task {task_id+1}: ",end="")
        for input, target, domain,class_group in data:
            domain = domain.to(device, non_blocking=True)[0].item() # torch CUDA-> integer
            class_group = class_group.to(device, non_blocking=True)[0].item()
            
            if task_id>0: # Define task type
                prev_domain, prev_class_group,_ = task_config[task_id-1]
                if prev_domain != domain and prev_class_group!= class_group:
                    task_type="CDIL"
                    cdil+=1
                elif prev_domain != domain:
                    task_type="DIL"
                    dil+=1
                elif prev_class_group != class_group:
                    task_type="CIL"
                    cil+=1
            else:
                task_type="Initial"
            task_config.append((domain, class_group,task_type))
            print(f"(Domain #{domain}, Class_group #{class_group}, {task_type})")
            break
    print(f"CIL: {cil/(args.num_tasks-1):.2f}, DIL: {dil/(args.num_tasks-1):.2f}, CDIL: {cdil/(args.num_tasks-1):.2f}")
    print(task_config)
    return task_config

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                      criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None):
    
    #if args.print_tasks:
    #    args.task_config = task_configuration(data_loader, device, args)
        
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
   
            
    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            if args.use_e_prompt:
                                model.module.e_prompt.prompt.grad.zero_()
                                model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            if args.use_e_prompt:
                                model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        if args.use_e_prompt:
                            model.module.e_prompt.prompt_key.grad.zero_()
                            model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        if args.use_e_prompt:
                            model.e_prompt.prompt_key.grad.zero_()
                            model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                       
                        optimizer.param_groups[0]['params'] = model.parameters()
        
         

        # Double check
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
                
        if lr_scheduler:
            lr_scheduler.step(epoch)


                   
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
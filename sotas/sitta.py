import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import logging
import cv2
import json
import numpy as np
import scipy
import math
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import time

def calculate_ratio(curr_pred_list):
    ratio_list = []
    for curr_pred in curr_pred_list:
        prob_curr = torch.softmax(curr_pred, 1)
        entropy_full = softmax_entropy(prob_curr, mode='binary', full=True)
        if torch.sum(prob_curr>=0.5) > 0:
            ratio = 0
            neg = entropy_full[prob_curr<0.5].mean().item()
            ratio += neg
            pos = entropy_full[prob_curr>=0.5].mean().item()
            ratio += pos
            ret = ratio / 2
        else:
            ret = entropy_full.mean().item()
        ratio_list.append(ret)
    return ratio_list

def update_and_predict(net_list, mean_list, std_list, val_imgs, checkpoint_list=None, rho=0.05):
    pred_list, conf_list, sharpness_list = [], [], []
    for idx in range(len(net_list)):
        update_stats(net_list[idx], mean_list[idx], std_list[idx])
        curr_time = time.time()
        pred, conf = make_prediction(net_list[idx], val_imgs)
        net_list[idx].eval()
        mask_pred = net_list[idx](val_imgs)
        mask_pred = torch.softmax(mask_pred,1)
        loss = softmax_entropy(mask_pred)
        loss.backward()

        for p in net_list[idx].parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    p.grad = torch.ones(p.grad.shape).cuda()

        state = {}
        with torch.no_grad():
            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in net_list[idx].parameters() if p.grad is not None]), p=2)
            scale = rho / (grad_norm + 1e-12)
            for n,p in net_list[idx].named_parameters():
                if p.grad is None: continue
                state[n] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        
        # Maximum prediction
        _, conf_max = make_prediction(net_list[idx], val_imgs)
        sharp = conf_max-conf
        
        # Reset weights 
        # net_list[idx].load_state_dict(torch.load('/data2/jianghao/TTA-MT/STTA/save_model/mms2d_unet/source-A-source-model-latest.pth'))
        net_list[idx].load_state_dict(torch.load(checkpoint_list))

        pred_list.append(pred)
        conf_list.append(conf)
        sharpness_list.append(sharp)

    return pred_list, conf_list, sharpness_list


def make_prediction(net, imgs):
    with torch.no_grad():
        pred = net(imgs)
        pred_prob = torch.softmax(pred,1)

        e = softmax_entropy(pred_prob, mode='binary')
        confidence = 1 - e 

        pred_mask = (pred_prob > 0.5).float()
        return pred.cpu(), confidence.item()

def softmax_entropy(x, mode='standard', full=False, eps=1e-10):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("Input contains NaN or Inf values")

    if mode == 'binary':
        ret = -x * torch.log2(x + eps) - (1 - x) * torch.log2(1 - x + eps)
        ret[x==0] = 0
        ret[x==1] = 0
    elif mode == 'standard':
        ret = -x * torch.log(x + eps) - (1 - x) * torch.log(1 - x + eps)
        ret[x==0] = 0
        ret[x==1] = 0

    if torch.isnan(ret).any() or torch.isinf(ret).any():
        raise ValueError("Output contains NaN or Inf values")

    if full:
        return ret
    else:
        return ret.mean()
    
def entropy( p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

# Manual record/update running mean
def get_stats(net):
    mean, var = [], []
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mean.append(m.running_mean.clone().detach())
            var.append(m.running_var.clone().detach())
    return mean, var

def update_stats(net, mean, var):
    count = 0
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.running_mean = mean[count].clone().detach().cuda()
            m.running_var = var[count].clone().detach().cuda()
            count += 1

class TTA(nn.Module):
    def __init__(self, model, repeat_num = 1, check_p = None,  steps=1, episodic=False):
        super().__init__()
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.check_p = check_p
        self.net_list = []
        for i in range(repeat_num):
            self.net_list.append(model)
        
        self.ori_mean_list, self.ori_std_list = [], []
        for idx, net in enumerate(self.net_list):
            ori_mean, ori_std = get_stats(net)
            self.ori_mean_list.append(ori_mean)
            self.ori_std_list.append(ori_std)
        
        # Tracker
        self.interp_num = 5
        self.step_size = 1 / self.interp_num
        self.val_mean_all_list, self.val_std_all_list = [], []


    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(1):
            self.x = x
            outputs = self.forward_and_adapt(x)
            
        return outputs
    
    @torch.enable_grad() 
    def forward_and_adapt(self, val_imgs):
        tmp_pred, tmp_conf, tmp_sharpness, tmp_score = [], [], [], []
        tmp_ratio = []

        val_mean_list, val_std_list = [], []

        for net_idx, net in enumerate(self.net_list):
            # Obtain val mean and std
            update_stats(net, self.ori_mean_list[net_idx], self.ori_std_list[net_idx])
            net.train()
            with torch.no_grad():
                net(val_imgs)
            val_mean, val_std = get_stats(net)
            net.eval()
            update_stats(net, self.ori_mean_list[net_idx], self.ori_std_list[net_idx])

            val_mean_list.append(val_mean)
            val_std_list.append(val_std)
        
        input_means = []
        input_stds =  []

        for i in range(self.interp_num+1):
            mix_avg_mean_list, mix_avg_std_list = [], []
            rate = i * self.step_size

            for idx in range(len(self.net_list)):
                tmp_mean = [(1-rate)*m1.cpu()+rate*m2.cpu() for m1,m2 in zip(self.ori_mean_list[idx], val_mean_list[idx])]
                tmp_std =  [(1-rate)*s1.cpu()+rate*s2.cpu() for s1,s2 in zip(self.ori_std_list[idx], val_std_list[idx])]
                mix_avg_mean_list.append(tmp_mean)
                mix_avg_std_list.append(tmp_std)
            input_means.append(mix_avg_mean_list)
            input_stds.append(mix_avg_std_list)

        # Run the prediction with different stats
        for stats_idx, (input_mean_list, input_std_list) in enumerate(zip(input_means, input_stds)):
            
            pred_list, conf_list, sharpness_list = update_and_predict(self.net_list, input_mean_list, input_std_list, self.x, \
                                                                                checkpoint_list=self.check_p, rho=0.1)
            ratio_list = calculate_ratio(pred_list)
            tmp_ratio.append(ratio_list)
            tmp_pred.append(torch.cat(pred_list, dim=0).unsqueeze(0))
            tmp_conf.append(conf_list)
            tmp_sharpness.append(sharpness_list)

        # return outout
        # Summary prediction
        # number of weight * number of model

        # N * num_network * shape
        tmp_pred_raw = torch.cat(tmp_pred)
        tmp_pred = torch.softmax(tmp_pred_raw, 2)
        
        # N * num_network
        tmp_conf  = torch.tensor(tmp_conf)
        tmp_score = torch.tensor(tmp_score)
        tmp_sharpness = torch.tensor(tmp_sharpness)

        # N * num_network, first column is dummy
        tmp_ratio = torch.tensor(tmp_ratio)
        
        # Simple average
        select_idx = len(tmp_pred)

        def get_score(entropy=False, sharpness=False, ratio=False, k=-1, normalize=False):
            weighted_score = []
            for pred_idx in range(len(self.net_list)):
                curr_net_pred = torch.clone(tmp_pred[:,pred_idx])
                if entropy:
                    weights = tmp_conf[:,pred_idx]
                    weight_format = 'entropy'
                elif ratio:
                    weights = -tmp_ratio[:,pred_idx]
                    weight_format = 'ratio'
                elif sharpness:
                    weights = tmp_sharpness[:,pred_idx]
                    weight_format = 'sharpness'
                else:
                    weights = None
                    weight_format = 'average'

                if normalize:  
                    weights = (weights - weights.min()) / (weights.max() - weights.min())
                if k > 0:
                    _, weighted_idx = torch.topk(weights, k=k)

                    weighted_pred = []
                    for wi in weighted_idx:
                        weighted_pred.append(curr_net_pred[wi].unsqueeze(0))
                    if normalize:
                        tmp = torch.cat(tmp, dim=0)
                        tmp_weight = torch.tensor(tmp_weight)
                        tmp_weight = (tmp_weight - tmp_weight.min()) / (tmp_weight.max() - tmp_weight.min())
                        tmp_weight = torch.softmax(tmp_weight / 1.0, dim=0)
                        weighted_pred = tmp.T @ tmp_weight
                    else:
                        weighted_pred = torch.cat(weighted_pred, dim=0).mean(0)
                    mask_pred = (weighted_pred > 0.5).float().unsqueeze(0).contiguous()
                else:
                    if weights is not None:
                        weights = torch.softmax(weights / 1.0, dim=0)
                        weighted_pred = curr_net_pred.T @ weights
                        # mask_pred = (weighted_pred.T > 0.5).float().unsqueeze(0).contiguous()
                        mask_pred = (weighted_pred.T).float().unsqueeze(0).contiguous()
                    else:
                        weighted_pred = (curr_net_pred>0.5).float().mean(0)
                        mask_pred = (weighted_pred > 0.5).float().unsqueeze(0).contiguous()

            return mask_pred
        


        output = get_score(sharpness=True, normalize=True)
        

        return output
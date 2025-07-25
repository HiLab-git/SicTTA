from copy import deepcopy
import torch
import torch.optim as optim
from sotas import (sictta, tent, norm, meant, sar, sitta)
from utils.sam import SAM
from utils.conf import cfg
import math

def setup_sar(model):
    model = sar.configure_model(model)
    params, param_names = sar.collect_params(model)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=0.01, momentum=0.9)
    adapt_model = sar.SAR(model, optimizer, margin_e0=0.4*math.log(4))
    return adapt_model

def setup_norm(model):
    norm_model = norm.Norm(model)
    stats, stat_names = norm.collect_stats(model)
    return norm_model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    return model,tent_model


def create_ema_model(model):
    ema_model = deepcopy(model) 

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def setup_meant(model):
    anchor_model = deepcopy(model)
    model.train()
    anchor_model.eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.OPTIM.LR,betas=(0.5,0.999))
    cotta_model = meant.TTA(model, anchor_model, optimizer,
                           mt_alpha=cfg.OPTIM.MT
                           )
    return cotta_model



def setup_sitta(model):
    cotta_model = sitta.TTA(model, 
                            repeat_num = 1,
                            check_p = cfg.MODEL.CKPT_DIR
                           )
    return cotta_model


def setup_sictta(model):
    anchor_model = deepcopy(model)
    model = sictta.configure_model(model)
    model.train()
    anchor_model.eval()
    
    sictta_model = sictta.TTA(model, anchor_model)
    return sictta_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError
from copy import deepcopy
import torch.nn.functional as F
import torchvision.transforms.functional as FF
import torch
import torch.nn as nn
import torch


class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, anchor_model,optimizer, mt_alpha=0.99):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.model_ema = anchor_model
        self.num_classes = 4 
        self.mt = mt_alpha
    def forward(self, x):
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    torch.autograd.set_detect_anomaly(True)
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        layer_fea = 'med'
        latent_model_ = model.get_feature(x, loc = layer_fea) 
        outputs = model.get_output(latent_model_,loc = layer_fea)
        standard_ema = self.model_ema(x)
        loss = (softmax_entropy(outputs, standard_ema)).mean(0) 
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        update_ema_variables(self.model_ema, self.model, self.mt)  # 直接调用

        # self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        return model(x)

def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    n, c, h, w =  x.shape
    entropy1 = -(x_ema.softmax(1) * x.log_softmax(1)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))
    return entropy1

    
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model
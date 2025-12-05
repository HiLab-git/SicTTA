from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTA(nn.Module):
    """Mean Teacher style TTA with on-line consistency regularization."""

    def __init__(self, model, anchor_model, optimizer, mt_alpha=0.99):
        super().__init__()
        self.student = model
        self.teacher = deepcopy(anchor_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.optimizer = optimizer
        self.momentum = mt_alpha

    def forward(self, x):
        return self.forward_and_adapt(x)

    def forward_and_adapt(self, x):
        self.student.train()
        outputs = self.student(x)

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        loss = consistency_loss(outputs, teacher_logits)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        update_ema(self.teacher, self.student, self.momentum)
        return outputs.detach()


def consistency_loss(student_logits, teacher_logits):
    """Spatially-averaged KL divergence between student and teacher outputs."""
    student_log_probs = F.log_softmax(student_logits, dim=1)
    teacher_probs = F.softmax(teacher_logits, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl


def update_ema(teacher, student, momentum):
    with torch.no_grad():
        for ema_param, param in zip(teacher.parameters(), student.parameters()):
            ema_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)
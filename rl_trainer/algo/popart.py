# from MAPPO https://github.com/marlbenchmark/on-policy

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter


class PopArt(nn.Module):

    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device="cpu"):

        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=torch.device(device))

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector, task_mask,dim=None):
        """
        input_vector: [T,R,N]
        task_mask: [T,R,N]
        """
        if dim is None:
            dim = self.norm_axes
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        task_mask = task_mask.to(**self.tpdv)
        mask_value = task_mask * input_vector

        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        # calculate the mean and sq mean
        n = task_mask.sum(dim=tuple(range(dim)))
        batch_mean = mask_value.sum(dim=tuple(range(dim))) / n.clamp(min=1e-10)
        batch_sq_mean = (mask_value ** 2).sum(dim=tuple(range(dim))) / n.clamp(min=1e-10)

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev = Parameter((self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4))

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        self.weight = Parameter((self.weight.t()* old_stddev / new_stddev).t())
        self.bias = Parameter((old_stddev * self.bias + old_mean - new_mean) / new_stddev)

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector,task_mask=None,dim=None):
        """
        Args:
            input_vector: [...]
            task_mask:[...,TASK_NUM]
        Returns:
            [...]
        """
        if dim is None:
            dim = self.norm_axes
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        task_mask = task_mask.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        if task_mask is None:
            mean = mean[(None,) * dim]
            std = torch.sqrt(var)[(None,) * dim]
        else:
            mean = (mean[(None,) * dim]*task_mask).sum(-1)
            std = (torch.sqrt(var)[(None,) * dim]*task_mask).sum(-1)

        out = (input_vector - mean) / std
        return out

    def denormalize(self, input_vector,task_mask=None,dim=None):
        """
        Args:
            input_vector: [...]
            task_mask:[...,TASK_NUM]
        Returns:
            [...]
        """
        if dim is None:
            dim = self.norm_axes
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()

        mean = mean[(None,) * dim]
        std = torch.sqrt(var)[(None,) * dim]

        out = input_vector * std + mean

        if task_mask is not None:
            out = (out*task_mask).sum(dim=dim)
        return out
    def record(self,writer:SummaryWriter):
        if writer is not None:
            T = self.mean.shape[0]
            for i in range(T):
                writer.add_scalar(f"popart_mean{i}",self.mean[i].item())
                writer.add_scalar(f"popart_std{i}",(self.stddev[i].item()))


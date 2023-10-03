import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

class DiffDriveDx(nn.Module):
    def __init__(self):
        super().__init__()

        self.dt = 0.25

        self.n_state = 3
        self.n_ctrl = 2

        self.goal_state = torch.Tensor([5., 5., 0.]) # corner, facing forward
        self.goal_weights = torch.Tensor([1., 1., 0.1]) # 0.1 weight for the angle
        self.ctrl_penalty = 0.0001 # 0.001 coefficient for ctrl in loss function

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        self.wheel_base = 1.
        self.wheel_radius = 0.5

    def forward(self, state, u):
        x, y, theta = torch.unbind(state, dim=1)
        l, r = torch.unbind(u, dim=1)
        state = torch.stack((
            self.wheel_radius / 2 * (l + r) * torch.cos(theta) * self.dt,
            self.wheel_radius / 2 * (l + r) * torch.sin(theta) * self.dt,
            self.wheel_radius / self.wheel_base * (r - l) * self.dt
        ), dim=1)
        return state

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        # p :: torch.Size([1200, 6])
        # q :: torch.Size([6])
        return Variable(q), Variable(p)

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class KBicycleDx(nn.Module):
    def __init__(self):
        super().__init__()

        self.dt = 0.05
        self.n_state = 5
        self.n_ctrl = 2 # acceleration and steering_angle

        self.max_torque = 2.
        self.L = 3.

        self.goal_state = torch.Tensor([0., 0., 0., 0., 0.])
        self.goal_weights = torch.Tensor([1., 1., 0.1, 0.1, 0.])
        self.ctrl_penalty = 0.001
        # bounds on acceleration and steering_angle - needs to be input as [T, n_batch, n_ctrl] into MPC
        self.lower, self.upper = torch.Tensor([-2., -np.pi]), torch.Tensor([2., np.pi])

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        # unlike the other envs, I don't need clamping
        # since all my architectures are forced to stay within the bounds
        # but I'll keep it anyway

        a, steering_angle = torch.unbind(u, dim=1)
        # a = torch.clamp(a, min=-2., max=2.)
        # steering_angle = torch.clamp(steering_angle, min=-np.pi, max=np.pi)
        x, y, cos_th, sin_th, v = torch.unbind(x, dim=1)
        theta = torch.atan2(sin_th, cos_th)

        x = x + (v*torch.cos(theta)) * self.dt
        y = y + (v*torch.sin(theta)) * self.dt
        theta = theta + (v*torch.tan(steering_angle)/self.L) * self.dt
        v = v + a * self.dt

        state = torch.stack((x, y, torch.cos(theta), torch.sin(theta), v), dim=1)

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        pass

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)

if __name__ == '__main__':
    pass
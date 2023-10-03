import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

class MountainCarDx(nn.Module):
    def __init__(self):
        super().__init__()

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        # self.goal_position = 0.5
        # self.goal_velocity = 0.

        self.dt = 0.05
        self.n_state = 2
        self.n_ctrl = 1
        self.goal_state = torch.Tensor([0.45, 0.])
        self.goal_weights = torch.Tensor([1., 0.1]) # 0.1 weight for the derivative
        self.ctrl_penalty = 0.1 # 0.1 coefficient for ctrl in loss function

        self.force = 0.001
        self.gravity = 0.0025

        self.low = torch.Tensor([self.min_position, -self.max_speed])
        self.high = torch.Tensor([self.max_position, self.max_speed])

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        """
        Based on https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/

        There are 3 actions:

        - -1: Accelerate to the left
        - 1: Accelerate to the right

        In other words, the continuous action range is (-1, 1)
        """
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        position, velocity = torch.unbind(x, dim=1)
        velocity = velocity + ((u * self.force).squeeze() + (torch.cos(3 * position) * (-self.gravity)))
        velocity = torch.clamp(velocity, -self.max_speed, self.max_speed)
        position = position + velocity
        position = torch.clamp(position, self.min_position, self.max_position)

        state = torch.stack((position, velocity), dim=1)

        if squeeze:
            state = state.squeeze(0)

        return state

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
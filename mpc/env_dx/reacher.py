import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

class ReacherDx(nn.Module):
    def __init__(self, num_joints=2):
        super().__init__()

        """Initialize Reacher environment

        Parameters
        ----------
        num_joints
                   Number of joints in reacher.

        """
        self.num_joints = num_joints
        self.torque_scale = 1.0
        self.dt = 0.05

        # self.action_space = gym.spaces.Box(
        #     low=-1.0, high=1.0, shape=[num_joints], dtype=np.float32
        # )

        self.n_state = 4
        self.n_ctrl = 2

        # self.goal_state = torch.Tensor([0., 0., 0., 0.]) # center
        self.goal_weights = torch.Tensor([1., 1., 0.1, 0.1]) # 0.1 weight for the derivative
        self.ctrl_penalty = 0.001 # 0.001 coefficient for ctrl in loss function

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        """

        Parameters
        ----------
        action :: (n_batch, 2)

        """
        B = x.shape[0]
        angle1, angle2, angle_vel1, angle_vel2 = torch.unbind(x, dim=1)
        angle_accs = self.torque_scale * u
        angle_vel1 = angle_vel1 + self.dt * angle_accs[:, 0]
        angle_vel2 = angle_vel2 + self.dt * angle_accs[:, 1]
        angle1 = angle1 + self.dt * angle_vel1
        angle2 = angle2 + self.dt * angle_vel2
        state = torch.stack((angle1, angle2, angle_vel1, angle_vel2), dim=1)
        return state

    def get_true_obj(self, goal_state):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        # px = -torch.sqrt(self.goal_weights)*self.goal_state
        # p = torch.cat((px, torch.zeros(self.n_ctrl)), dim=1)
        B = goal_state.shape[0]
        goal_weights = self.goal_weights.unsqueeze(0).repeat(B, 1)
        px = -torch.sqrt(goal_weights)*goal_state
        p = torch.cat((px, torch.zeros((B, self.n_ctrl))), dim=1)
        # p :: torch.Size([1200, 6])
        # q :: torch.Size([6])
        return Variable(q), Variable(p)

import torch

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost
from mpc.dynamics import NNDynamics
import mpc.util as eutil
# from mpc.env_dx import pendulum, cartpole
from mpc.env_dx import pendulum, cartpole, kbicycle, mountaincar, reacher, diffdrive

import numpy as np
import numpy.random as npr

import os
import sys
import shutil
import time

import pickle as pkl

#from setproctitle import setproctitle

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

class IL_Env:
    def __init__(self, env, lqr_iter=500, mpc_T=20, slew_rate_penalty=None, u_upper=2., u_lower=-2.):
        self.env = env

        if self.env == 'pendulum':
            self.true_dx = pendulum.PendulumDx()
        elif self.env == 'cartpole':
            self.true_dx = cartpole.CartpoleDx()
        elif self.env == 'pendulum-complex':
            params = torch.tensor((10., 1., 1., 1.0, 0.1))
            self.true_dx = pendulum.PendulumDx(params, simple=False)
        elif self.env == "kbicycle":
            self.true_dx = kbicycle.KBicycleDx()
        elif self.env == "mountaincar":
            self.true_dx = mountaincar.MountainCarDx()
        elif self.env == "reacher":
            self.true_dx = reacher.ReacherDx()
        elif self.env == "diffdrive":
            self.true_dx = diffdrive.DiffDriveDx()
        else:
            assert False

        if u_upper:
            self.u_upper = u_upper
        else:
            self.u_upper = self.true_dx.upper

        if u_lower:
            self.u_lower = u_lower
        else:
            self.u_lower = self.true_dx.lower

        self.lqr_iter = lqr_iter
        self.mpc_T = mpc_T
        self.slew_rate_penalty = slew_rate_penalty

        self.grad_method = GradMethods.AUTO_DIFF

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def sample_xinit(self, n_batch=1):
        if self.env in ['pendulum', 'pendulum-complex']:
            th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
            thdot = uniform(n_batch, -1., 1.)
            xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)
        elif self.env == 'cartpole':
            # qpos = uniform((self.n_batch, 2), -0.1, 0.1)
            # qvel = uniform((self.n_batch, 2), -0.1, 0.1)
            # xinit = torch.cat((qpos, qvel), dim=1)
            x = uniform(n_batch, -0.5, 0.5)
            dx = uniform(n_batch, -0.5, 0.5)
            th = uniform(n_batch, -np.pi, np.pi)
            dth = uniform(n_batch, -1., 1.)
            xinit = torch.stack((x, dx, torch.cos(th), torch.sin(th), dth), dim=1)
        elif self.env == "kbicycle":
            x = uniform(n_batch, -1., 1.)
            y = uniform(n_batch, -1., 1.)
            th = uniform(n_batch, -np.pi, np.pi)
            v = uniform(n_batch, -1., 1.)
            xinit = torch.stack((x, y, torch.cos(th), torch.sin(th), v), dim=1)
        elif self.env == "mountaincar":
            p = uniform(n_batch, -0.6, -0.4)
            v = torch.zeros(n_batch)
            xinit = torch.stack((p, v), dim=1)
        elif self.env == "reacher":
            angle1 = uniform(n_batch, -np.pi, np.pi)
            angle2 = uniform(n_batch, -np.pi, np.pi)
            angle_vel1 = uniform(n_batch, -1, 1)
            angle_vel2 = uniform(n_batch, -1, 1)
            xinit = torch.stack((angle1, angle2, angle_vel1, angle_vel2), axis=1)
        elif self.env == "diffdrive":
            x = uniform(n_batch, -5, 5)
            y = uniform(n_batch, -5, 5)
            theta = uniform(n_batch, -2*torch.pi, 2*torch.pi)
            xinit = torch.stack((x, y, theta), axis=1)
        else:
            import ipdb; ipdb.set_trace()

        return xinit

    def populate_data(self, n_train, n_val, n_test, seed=0):
        torch.manual_seed(seed)

        n_data = n_train+n_val+n_test
        xinit = self.sample_xinit(n_batch=n_data)

        if self.env == "reacher":
            n_batch = xinit.shape[0]
            goal_xy = uniform((n_batch, 2), -2, 2)
            goal_state = torch.cat((goal_xy, torch.zeros((n_batch, 2))), dim=1)
            true_q, true_p = self.true_dx.get_true_obj(goal_state)

            self.train_goals = goal_state[:n_train]
            self.val_goals = goal_state[n_train:n_train+n_val]
            self.test_goals = goal_state[-n_test:]
        else:
            true_q, true_p = self.true_dx.get_true_obj()

        true_x_mpc, true_u_mpc = self.mpc(self.true_dx, xinit, true_q, true_p)
        tau = torch.cat((true_x_mpc, true_u_mpc), dim=2).transpose(0,1)

        self.train_data = tau[:n_train]
        self.val_data = tau[n_train:n_train+n_val]
        self.test_data = tau[-n_test:]

    def mpc(self, dx, xinit, q, p, u_init=None, eps_override=None,
            lqr_iter_override=None):
        n_batch = xinit.shape[0]

        n_sc = self.true_dx.n_state+self.true_dx.n_ctrl

        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
            self.mpc_T, n_batch, 1, 1
        )
        if self.env == "reacher":
            p = p.unsqueeze(0).repeat(self.mpc_T, 1, 1)
        else:
            p = p.unsqueeze(0).repeat(self.mpc_T, n_batch, 1)

        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter

        x_mpc, u_mpc, objs_mpc = mpc.MPC(
            self.true_dx.n_state, self.true_dx.n_ctrl, self.mpc_T,
            u_lower=self.u_lower, u_upper=self.u_upper, u_init=u_init,
            lqr_iter=lqr_iter,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=True,
            linesearch_decay=self.true_dx.linesearch_decay,
            max_linesearch_iter=self.true_dx.max_linesearch_iter,
            grad_method=self.grad_method,
            eps=eps,
            # slew_rate_penalty=self.slew_rate_penalty,
            # prev_ctrl=prev_ctrl,
        )(xinit, QuadCost(Q, p), dx)
        return x_mpc, u_mpc


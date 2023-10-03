from tqdm import tqdm

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset, DataLoader

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx
from mpc.dynamics import NNDynamics
import mpc.util as eutil
from mpc.env_dx import pendulum, cartpole

import numpy as np
import numpy.random as npr

import argparse
import os
import sys
import shutil
import time
import re

import pickle as pkl

from il_env import IL_Env

import time

class ReflexNet(nn.Module):
    def __init__(self, n_state, n_ctrl, h_dim, T=20):
        super(ReflexNet, self).__init__()

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T

        self.net = nn.Sequential(
            nn.Linear(n_state, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim*2),
            nn.ReLU(),
            nn.Linear(h_dim*2, h_dim*4),
            nn.ReLU(),
            nn.Linear(h_dim*4, T*n_ctrl),
            nn.Tanh()
        )

    def forward(self, xinits, scalar=2.):
        return (self.net(xinits) * scalar)

class RNN(nn.Module):
    def __init__(self, n_state, n_ctrl, h_dim, T=20):
        super(RNN, self).__init__()

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T

        self.state_emb = nn.Sequential(
            nn.Linear(n_state, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.ctrl_emb = nn.Sequential(
            nn.Linear(n_ctrl, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.decode = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_ctrl),
            nn.Tanh()
        )
        self.cell = nn.LSTMCell(h_dim, h_dim)

    def forward(self, xinits, scalar=2.):
        yt = self.state_emb(xinits)
        cell_state = None
        uts = []

        for t in range(self.T):
            cell_state = self.cell(yt, cell_state)
            ht, ct = cell_state
#             ut = self.decode(ct)
            ut = self.decode(ct) * scalar
            uts.append(ut)
            yt = self.ctrl_emb(ut)

        uts = torch.stack(uts, dim=1)
        return uts

class KoopmanNet(nn.Module):
    def __init__(self, dt, n_state, n_ctrl, h_dim, z_dim, aux_dim):
        super(KoopmanNet, self).__init__()
        
        self.dt = dt
        self.z_dim = z_dim
        self.n_state = n_state
        self.n_ctrl = n_ctrl

        self.encoder = nn.Sequential(nn.Linear(n_state, h_dim),
#                                      nn.GELU(),
#                                      nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, z_dim))
        
        self.decoder = nn.Sequential(nn.Linear(z_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, n_state))
        
        # aux
        self.aux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim))
        
        self.bux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim))
        
        self.diag_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim+n_ctrl))
        
        self.lower_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, ((z_dim+n_ctrl)**2)//2 - z_dim+n_ctrl))
        
        self.pnn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim+n_ctrl))

    def block(self, a, b):
        return torch.exp(a*dt) * torch.stack((torch.stack((torch.cos(b*dt), -torch.sin(b*dt))),
                                              torch.stack((torch.sin(b*dt), torch.cos(b*dt)))))

    def aux(self, x):
        a, b = self.aux_nn(x)
        return self.block(a, b)

    def bux(self, x):
        a, b = self.bux_nn(x)
        return torch.stack((a.unsqueeze(0), b.unsqueeze(0)))

    def psd(self, diag, lower):
        L = torch.diag(diag)
        idx1, idx2 = np.tril_indices(len(L), k=-1)
        L = L.index_put((torch.tensor(idx1), torch.tensor(idx2)), lower)
        return L

    def cux(self, x):
        diag = self.diag_nn(x)
        lower = self.lower_nn(x)

        L = self.psd(diag, lower)
        C = L @ L.T
        return C
    
    def pux(self, x):
        return self.pnn(x)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def identity(self, x):
        return self.decode(self.encode(x))

    def predict(self, z, u, x):
        return self.aux(x) @ z + self.bux(x) @ u

class BlockKoopmanNet(nn.Module):
    def __init__(self, dt, n_state, n_ctrl, h_dim, z_dim, aux_dim):
        super(BlockKoopmanNet, self).__init__()
        
        self.dt = dt
        self.z_dim = z_dim
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        
        self.n_block = z_dim // 2 # number of complex-conjugate pair Jordan blocks
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(nn.Linear(n_state, h_dim),
#                                      nn.GELU(),
#                                      nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, z_dim))
        
        self.decoder = nn.Sequential(nn.Linear(z_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, n_state))

        # aux
        self.aux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim))
        
        self.bux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim*n_ctrl))
        
        self.diag_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim+n_ctrl))
        
        self.lower_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, ((z_dim+n_ctrl)**2 - (z_dim+n_ctrl))//2)) # subtract the diagonal

        self.pnn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim+n_ctrl))

    def block(self, a, b):
        return torch.exp(a*dt) * torch.stack((torch.stack((torch.cos(b*dt), -torch.sin(b*dt))),
                                              torch.stack((torch.sin(b*dt), torch.cos(b*dt)))))

    def aux(self, x):
        A = []
        for a, b in torch.tensor_split(self.aux_nn(x), self.n_block):
            A.append(self.block(a, b))
        return torch.block_diag(*A)

    def bux(self, x):
        return torch.reshape(self.bux_nn(x), (self.z_dim, self.n_ctrl))

    def psd(self, diag, lower):
        L = torch.diag(diag)
        idx1, idx2 = np.tril_indices(len(L), k=-1)
        L = L.index_put((torch.tensor(idx1), torch.tensor(idx2)), lower)
        return L

    def cux(self, x):
        diag = self.diag_nn(x)
        lower = self.lower_nn(x)

        L = self.psd(diag, lower)
        C = L @ L.T
        return C
    
    def pux(self, x):
        return self.pnn(x)

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def identity(self, x):
        return self.decode(self.encode(x))

    def predict(self, z, u, x):
        return self.aux(x) @ z + self.bux(x) @ u

class ContinuousYinNetwork(nn.Module):
    def __init__(self, dt, n_state, n_ctrl, h_dim, z_dim, aux_dim, T=10, scalar=2.):
        super(ContinuousYinNetwork, self).__init__()

        self.dt = dt
        self.z_dim = z_dim
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        
        self.n_block = z_dim // 2 # number of complex-conjugate pair Jordan blocks
        self.z_dim = z_dim

        self.T = T

        self.xr = nn.Parameter(torch.zeros(z_dim,))
        self.scalar = scalar

        self.encoder = nn.Sequential(nn.Linear(n_state, h_dim),
#                                      nn.GELU(),
#                                      nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, z_dim))
        
        self.decoder = nn.Sequential(nn.Linear(z_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.GELU(),
                                     nn.Linear(h_dim, n_state))

        # aux
        self.aux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim))
        
        self.bux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim*n_ctrl))
        
        self.diag_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, z_dim))
        
        self.lower_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, ((z_dim)**2)//2 - z_dim//2)) # subtract the diagonal
        
        self.rux_nn = nn.Sequential(nn.Linear(n_state, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, aux_dim),
                                 nn.GELU(),
                                 nn.Linear(aux_dim, n_ctrl*n_ctrl))

    def block(self, a, b):
        return torch.exp(a*dt) * torch.stack((torch.stack((torch.cos(b*dt), -torch.sin(b*dt))),
                                              torch.stack((torch.sin(b*dt), torch.cos(b*dt)))))

    def aux(self, x):
        A = []
        for a, b in torch.tensor_split(self.aux_nn(x), self.n_block):
            A.append(self.block(a, b))
        return torch.block_diag(*A)

    def bux(self, x):
        return torch.reshape(self.bux_nn(x), (self.z_dim, self.n_ctrl))

    def psd(self, diag, lower):
        L = torch.diag(diag)
        idx1, idx2 = np.tril_indices(len(L), k=-1)
        L = L.index_put((torch.tensor(idx1), torch.tensor(idx2)), lower)
        return L

    def qux(self, x):
        # obtain positive semidefinite Q
        diag = self.diag_nn(x)
        lower = self.lower_nn(x)
        
        L = self.psd(diag, lower)
        C = L @ L.T
        return C
    
    def rux(self, x):
        # obtain positive definite R
        M, _ = torch.linalg.qr(torch.reshape(self.rux_nn(x), (self.n_ctrl, self.n_ctrl))) # use QR decomposition to obtain a full rank matrix
        return M.T @ M

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def identity(self, x):
        return self.decode(self.encode(x))

    def predict(self, z, u, x):
        return self.aux(x) @ z + self.bux(x) @ u

    def dare(self, A, B, Q, R, N=150): # play with N, ranges from 25 to 150
        """
        Solve the discrete algebraic Riccati equation (DARE)
        """
        P = torch.eye(A.size(0))
        for _ in range(N):
            next_P = Q + A.T @ P @ A - A.T @ P @ B @ torch.inverse(R + B.T @ P @ B) @ B.T @ P @ A
            P = next_P
        return P

    def lqr(self, x0, A, B, Q, R):
        """
        Solve the discrete time LQR controller.
        """
        P = self.dare(A, B, Q, R)

        # Compute the LQR gain
        K = -torch.inverse(R + B.T @ P @ B) @ (B.T @ P @ A)

        x = x0
        x_traj = [x0]
        u_traj = []

        for _ in range(self.T):
            u = K @ (x - self.xr)
            x = A @ x + B @ u

            x_traj.append(x)
            u_traj.append(u)
        
        return torch.stack(x_traj), torch.tanh(torch.stack(u_traj)) * self.scalar

def predict(model, zinit, u, xinit):
    z_next = zinit
    x_nexts = []
    z_nexts = []
    for i in range(T):
        z_next = model.predict(z_next, u[i], xinit)
        x_nexts.append(model.decode(z_next))
        z_nexts.append(z_next)
    return torch.vstack(x_nexts), torch.vstack(z_nexts)

def pred_loss_fn(model, zinits, xinits, x_true, u_true):
    z_true = torch.vmap(model.encode)(x_true)
    x_pred, z_pred = torch.vmap(predict, in_dims=(None, 0, 0, 0))(model, zinits, u_true, xinits)
    pred_loss = torch.mean((x_pred - x_true)**2) + torch.mean((z_pred - z_true)**2)
    return pred_loss

def loss_fn(model, xinits, x_true, u_true, u_upper, u_lower, T=20, is_pred_loss=False, is_constrained=True):
    batch_size = xinits.shape[0]
    
    zinits = torch.vmap(model.encode)(xinits)
    dx = LinDx(torch.cat((torch.vmap(model.aux)(xinits), torch.vmap(model.bux)(xinits)), dim=-1).unsqueeze(0).repeat(T, 1, 1, 1))
    C = torch.vmap(model.cux)(xinits).unsqueeze(0).repeat(T, 1, 1, 1)
    c = torch.vmap(model.pux)(xinits).unsqueeze(0).repeat(T, 1, 1)
    cost = QuadCost(C, c)

    u_init = None

    z_pred, u_pred, objs_pred = mpc.MPC(
        model.z_dim, n_ctrl, T,
        u_lower=u_lower, u_upper=u_upper, u_init=u_init,
        lqr_iter=T*2 if is_constrained else 1,
        verbose=-1,
        exit_unconverged=False,
        detach_unconverged=False,
        n_batch=batch_size,
    )(zinits, cost, dx)

    # note that I'm not warm-starting, I could use the indices for that
    
    x_pred = torch.vmap(model.decode)(z_pred)

    u_pred = u_pred.transpose(0, 1) # (T, B, 1) -> (B, T, 1) to match u_true's shape
    x_pred = x_pred.transpose(0, 1) # (T, B, 3) -> (B, T, 3) to match x_true's shape

    im_loss = torch.mean((u_true - u_pred)**2)
    sysid_loss = torch.mean((x_true - x_pred)**2)
    id_loss = torch.mean((model.identity(x_true) - x_true)**2)
    
    if is_pred_loss:
        pred_loss = pred_loss_fn(model, zinits, xinits, x_true, u_true)
    else:
        pred_loss = 0
    traj_loss = im_loss + id_loss

    return traj_loss, im_loss, id_loss

def loss_fn_rnn(model, xinits, x_true, u_true, T=20, scalar=None):
    batch_size = xinits.shape[0]

    if scalar:
        u_pred = model(xinits, scalar)
    else:
        u_pred = model(xinits)
    
    im_loss = torch.mean((u_true - u_pred)**2)
    sysid_loss = 0
    id_loss = 0
    pred_loss = 0
    traj_loss = im_loss

    return traj_loss, im_loss, id_loss#, sysid_loss, pred_loss

def loss_fn_reflex(model, xinits, x_true, u_true, T=20, scalar=None):
    batch_size = xinits.shape[0]

    if scalar:
        u_pred = model(xinits, scalar)
    else:
        u_pred = model(xinits)

    u_pred = u_pred.unsqueeze(-1)

    im_loss = torch.mean((u_true - u_pred)**2)
    id_loss = 0
    pred_loss = 0
    traj_loss = im_loss

    return traj_loss, im_loss, id_loss

def loss_fn_contyin(model, xinits, x_true, u_true, u_upper, u_lower, T=20, is_pred_loss=False, is_constrained=True):
    batch_size = xinits.shape[0]

    zinits = torch.vmap(model.encode)(xinits)
    A = torch.vmap(model.aux)(xinits)
    B = torch.vmap(model.bux)(xinits)
    Q = torch.vmap(model.qux)(xinits)
    R = torch.vmap(model.rux)(xinits)

    model.scalar = u_upper # attach scalar to the model
    z_pred, u_pred = torch.vmap(model.lqr)(zinits, A, B, Q, R)

    # note that I'm not warm-starting, I could use the indices for that

    im_loss = torch.mean((u_true - u_pred)**2)
    id_loss = torch.mean((model.identity(x_true) - x_true)**2)

    if is_pred_loss:
        pred_loss = pred_loss_fn(model, zinits, xinits, x_true, u_true)
    else:
        pred_loss = 0
    traj_loss = im_loss + id_loss

    return traj_loss, im_loss, id_loss

parser = argparse.ArgumentParser(prog="Koopman DMPC",
                                 description="Learn constrained policies.")
parser.add_argument("-en", "--env_name", default="pendulum")
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-p", "--project", default="koopman-dmpc")
parser.add_argument("-mt", "--model_type", default="koopman")
parser.add_argument("-e", "--num_epochs", default=250, type=int)
parser.add_argument("-b", "--batch_size", default=100, type=int)
parser.add_argument("-t", "--horizon", default=10, type=int)
parser.add_argument("-ub", "--upper_bound", default=2.0, type=float)
parser.add_argument("-lb", "--lower_bound", default=-2.0, type=float)
parser.add_argument("-ubt", "--upper_bound_test", default=1.0, type=float)
parser.add_argument("-lbt", "--lower_bound_test", default=-1.0, type=float)
args = parser.parse_args()

seed = args.seed

np.random.seed(seed)
batch_size = args.batch_size

def make_data(n_state, n_ctrl, data, batch_size=32, warmstart=None, shuffle=False):
    xs, us = data[:,:,:n_state], data[:,:,-n_ctrl:]
    xinits = xs[:,0]
    n_data = xinits.shape[0]
    ds = TensorDataset(xinits, xs, us, torch.arange(0,n_data))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return ds, loader

## assume data has already been created by make_data.py before reading into memory!

torch.manual_seed(seed)

env_name = args.env_name

u_upper = args.upper_bound
u_lower = args.lower_bound
u_upper_test = args.upper_bound_test
u_lower_test = args.lower_bound_test

with open("./data/{}_upper{:.1f}_lower{:.1f}.pkl".format(env_name, u_upper, u_lower), "rb") as f:
    env2 = pkl.load(f)

with open("./data/{}_upper{:.1f}_lower{:.1f}.pkl".format(env_name, u_upper_test, u_lower_test), "rb") as f:
    env1 = pkl.load(f)

n_state, n_ctrl = env2.true_dx.n_state, env2.true_dx.n_ctrl

# train bounds
train_data, train = make_data(n_state, n_ctrl,
                              env2.train_data, batch_size=batch_size, shuffle=True)
val_data2, val2 = make_data(n_state, n_ctrl,
                            env2.val_data, batch_size=batch_size)
test_data2, test2 = make_data(n_state, n_ctrl,
                              env2.test_data, batch_size=batch_size)

# test bounds
val_data1, val1 = make_data(n_state, n_ctrl,
                            env1.val_data, batch_size=batch_size)
test_data1, test1 = make_data(n_state, n_ctrl,
                              env1.test_data, batch_size=batch_size)

model_type = args.model_type

T = args.horizon
dt = 0.05

is_koopman = model_type == "koopman"
is_rnn = model_type == "rnn"
is_reflex = model_type == "reflex"
is_cyin = model_type == "cyin"

if is_koopman:
    model = BlockKoopmanNet(dt, n_state, n_ctrl, 80, 2 if env_name == "pendulum" or "mountaincar" else 6, 170)
elif is_rnn:
    model = RNN(n_state, n_ctrl, 256, T=T)
elif is_reflex:
    model = ReflexNet(n_state, n_ctrl, 256, T=T)
elif is_cyin:
    model = ContinuousYinNetwork(dt, n_state, n_ctrl, 80, 2 if env_name == "pendulum" or "mountaincar" else 6, 170, T=T)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = args.num_epochs
times = []
for epoch in tqdm(range(10)):
    start = time.time()
    for j, (xinits, x_true, u_true, idxs) in enumerate(train):

        if is_koopman:
            loss, im_loss, id_loss = loss_fn(model, xinits, x_true, u_true, u_upper, u_lower, T=T)
        elif is_rnn:
            loss, im_loss, id_loss = loss_fn_rnn(model, xinits, x_true, u_true, T=T, scalar=u_upper)
        elif is_reflex:
            loss, im_loss, id_loss = loss_fn_reflex(model, xinits, x_true, u_true, T=T, scalar=u_upper)
        elif is_cyin:
            loss, im_loss, id_loss = loss_fn_contyin(model, xinits, x_true, u_true, u_upper, u_lower, T=T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    times.append(time.time() - start)
print(times)
print(np.median(times), np.std(times))

with torch.no_grad():
    def no_grad(loader, u_upper, u_lower):
        avg_loss = []
        avg_im_loss = []
        avg_id_loss = []
        for j, (xinits, x_true, u_true, idxs) in enumerate(loader):
            if is_koopman:
                loss, im_loss, id_loss = loss_fn(model, xinits, x_true, u_true, u_upper, u_lower, T=T)
            elif is_rnn:
                loss, im_loss, id_loss = loss_fn_rnn(model, xinits, x_true, u_true, T=T, scalar=u_upper)
            elif is_reflex:
                loss, im_loss, id_loss = loss_fn_reflex(model, xinits, x_true, u_true, T=T, scalar=u_upper)
            elif is_cyin:
                loss, im_loss, id_loss = loss_fn_contyin(model, xinits, x_true, u_true, u_upper, u_lower, T=T)
            avg_loss.append(loss)
            avg_im_loss.append(im_loss)
            avg_id_loss.append(id_loss)
        return np.mean(avg_loss), np.mean(avg_im_loss), np.mean(avg_id_loss)

    ## (-2, 2)
    times = []
    for _ in range(10):
        start = time.time()
        avg_loss, avg_im_loss, avg_id_loss = no_grad(test2, u_upper_test, u_lower_test)
        times.append(time.time() - start)

print(times)
print(np.median(times), np.std(times))

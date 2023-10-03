import numpy as np
import pickle as pkl
import os
import sys
from il_env import IL_Env
import argparse

parser = argparse.ArgumentParser(prog="Data maker",
                                 description="Make data.")
parser.add_argument("-t", "--horizon", default=10, type=int)
parser.add_argument("-ub", "--upper_bound", default=2.0, type=float)
parser.add_argument("-lb", "--lower_bound", default=-2.0, type=float)
parser.add_argument("-ubt", "--upper_bound_test", default=1.0, type=float)
parser.add_argument("-lbt", "--lower_bound_test", default=-1.0, type=float)
parser.add_argument("-en", "--env_name", default="pendulum") # pendulum or cartpole
args = parser.parse_args()

np.random.seed(0)

def gen_data(u_upper, u_lower, mpc_T, env_name):
    data_dir = "./data/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    n_train, n_val, n_test = 1000, 100, 100
    env = IL_Env(env_name, lqr_iter=500, mpc_T=mpc_T, u_upper=u_upper, u_lower=u_lower)
    env.populate_data(n_train=n_train, n_val=n_val, n_test=n_test, seed=0)

    fname = env_name+'_upper{:.1f}_lower{:.1f}'.format(u_upper, u_lower)
    save = os.path.join(data_dir, fname+'.pkl')
    print('Saving data to {}'.format(save))
    with open(save, 'wb') as f:
        pkl.dump(env, f)

env_name = args.env_name

mpc_T = args.horizon

u_upper = args.upper_bound
u_lower = args.lower_bound

gen_data(u_upper, u_lower, mpc_T, env_name)

u_upper = args.upper_bound_test
u_lower = args.lower_bound_test

gen_data(u_upper, u_lower, mpc_T, env_name)

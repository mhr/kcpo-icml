import argparse
import os

# model_type = "koopman"
# model_type = "rnn"
model_type = "reflex"

# env_name = "pendulum"
env_name = "cartpole"

T = 10
# T = 20

parser = argparse.ArgumentParser(prog="Main runner", description="Run experiments from command line")
parser.add_argument("-t", "--horizon", default=10, type=int)
parser.add_argument("-mt", "--model_type", default="koopman")
parser.add_argument("-en", "--env_name", default="pendulum") # pendulum or cartpole
parser.add_argument("-ub", "--upper_bound", default=2.0, type=float)
parser.add_argument("-lb", "--lower_bound", default=-2.0, type=float)
parser.add_argument("-ubt", "--upper_bound_test", default=1.0, type=float)
parser.add_argument("-lbt", "--lower_bound_test", default=-1.0, type=float)
args = parser.parse_args()

model_type = args.model_type
# model_type = "koopman"
# model_type = "rnn"
# model_type = "reflex"

env_name = args.env_name
# env_name = "pendulum"
# env_name = "cartpole"

T = args.horizon

i = 0
print("================== BEGIN {} EXPERIMENT SEED={} ==================".format(model_type, i))
os.system("python timer.py --seed {} --model_type {} --env_name {} --horizon {} --upper_bound {} --lower_bound {} --upper_bound_test {} --lower_bound_test {}".format(
            i, 
            model_type,
            env_name,
            T,
            args.upper_bound,
            args.lower_bound,
            args.upper_bound_test,
            args.lower_bound_test))
import os
import argparse

parser = argparse.ArgumentParser(prog="Main runner", description="Run experiments from command line")
parser.add_argument("-t", "--horizon", default=10, type=int)
parser.add_argument("-mt", "--model_type", default="koopman")
parser.add_argument("-en", "--env_name", default="pendulum") # pendulum or cartpole
parser.add_argument("-ub", "--upper_bound", default=2.0, type=float)
parser.add_argument("-lb", "--lower_bound", default=-2.0, type=float)
parser.add_argument("-ubt", "--upper_bound_test", default=1.0, type=float)
parser.add_argument("-lbt", "--lower_bound_test", default=-1.0, type=float)
parser.add_argument("-cp", "--checkpoint", action="store_true")
args = parser.parse_args()

model_type = args.model_type

env_name = args.env_name

is_checkpoint = args.checkpoint

T = args.horizon

mainpy = "reacher_main.py" if env_name == "reacher" else "main.py"

if is_checkpoint:
    os.system("python {} --seed 0 --checkpoint --model_type {} --env_name {} --horizon {} --upper_bound {} --lower_bound {} --upper_bound_test {} --lower_bound_test {}".format(
                mainpy,
                model_type,
                env_name,
                T,
                args.upper_bound,
                args.lower_bound,
                args.upper_bound_test,
                args.lower_bound_test))
else:
    # 10 trials
    for i in range(10):
        print("================== BEGIN {} EXPERIMENT SEED={} UB={} LB={} UBT={} LBT={} ==================".format(model_type,
                                                                                                                   i,
                                                                                                                   args.upper_bound,
                                                                                                                   args.lower_bound,
                                                                                                                   args.upper_bound_test,
                                                                                                                   args.lower_bound_test))
        os.system("python {} --seed {} --model_type {} --env_name {} --horizon {} --upper_bound {} --lower_bound {} --upper_bound_test {} --lower_bound_test {}".format(
            mainpy,
            i,
            model_type,
            env_name,
            T,
            args.upper_bound,
            args.lower_bound,
            args.upper_bound_test,
            args.lower_bound_test))

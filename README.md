# README

## Installation
Try to run the Python scripts below and install things whenever things break.
- PyTorch 1.13.1 (CPU)
- tqdm
- numpy
- functorch (should be installed with PyTorch 1.13.1 automatically, but if you're getting an error, this might be it)

## Generating data
`$ python make_data.py --env_name pendulum --upper_bound 2.0 --lower_bound -2.0 --upper_bound_test 1.0 --lower_bound_test -1.0`

`$ python make_data.py --env_name cartpole --upper_bound 10 --lower_bound -10 --upper_bound_test 5 --lower_bound_test -5`

`$ python make_data.py --env_name mountaincar --upper_bound 1 --lower_bound -1 --upper_bound_test 0.5 --lower_bound_test -0.5`

`$ python make_data.py --env_name reacher --upper_bound 1 --lower_bound -1 --upper_bound_test 0.5 --lower_bound_test -0.5`

`$ python make_data.py --env_name diffdrive --upper_bound 100 --lower_bound -100 --upper_bound_test 80 --lower_bound_test -80`

## Training
`$ python seed_exp.py --model_type koopman --env_name pendulum --upper_bound 2.0 --lower_bound -2.0 --upper_bound_test 1.0 --lower_bound_test -1.0`

`$ python seed_exp.py --model_type koopman --env_name cartpole --upper_bound 10 --lower_bound -10 --upper_bound_test 5 --lower_bound_test -5`

`$ python seed_exp.py --model_type koopman --env_name reacher --upper_bound 1 --lower_bound -1 --upper_bound_test 0.5 --lower_bound_test -0.5`

`$ python seed_exp.py --model_type koopman --env_name diffdrive --upper_bound 100 --lower_bound -100 --upper_bound_test 80 --lower_bound_test -80`

"koopman" could be "reflex" or "rnn" (the baselines) too

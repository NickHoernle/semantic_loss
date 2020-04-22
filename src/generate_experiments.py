#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/robotic_constraints'
base_call = (f"python main_experiment_avoid.py --input {DATA_HOME}/data --output {DATA_HOME}/output "
             "--use_cuda True --batch_size 256 --num_epochs 2500 --early-stopping-lim 50 --num_workers 8 "
             "--max_grad_norm 1")

repeats = 1
learning_rates = [1e-2, 1e-3, 1e-4]
hidden_dim = [40, 50, 60, 70, 80]
gammas = [.9, .95, .99, .999]
backward = [True, False]
back_strength = [1e2, 1e3, 1e4, 1e5]
# num_layers = [40]
# gammas = [.99]

settings = [(lr, gam, h_dim, bk, back_, rep) for lr in learning_rates for gam in gammas for h_dim in hidden_dim for bk in back_strength for back_ in backward
            for rep in range(repeats)]
nr_expts = len(learning_rates) * len(hidden_dim) * len(gammas) * len(backward) * len(back_strength) * repeats

nr_servers = 15
avg_expt_time = 5  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, gam, h_dim, bk, back_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--gamma {gam} "
        f"--hidden_dim {h_dim} "
        f"--back_strength {bk} "
        f"--backward {back_}"
    )
    print(expt_call, file=output_file)

output_file.close()
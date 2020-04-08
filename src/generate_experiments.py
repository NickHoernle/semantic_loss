#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/robotic_constraints/'
base_call = (f"python main_experiment_avoid.py --input {DATA_HOME}/data --output {DATA_HOME}/output "
             "--use_cuda True --batch_size 256 --num_epochs 2000 --early-stopping-lim 250")

repeats = 3
learning_rates = [1e-2, 1e-3, 1e-4]
num_layers = [10, 20, 30, 40]
gammas = [.6, .7, .8, .9, .99]

settings = [(lr, gam, rep, lay) for lr in learning_rates for gam in gammas for lay in num_layers
            for rep in range(repeats)]
nr_expts = len(learning_rates) * len(gammas) * len(num_layers) * repeats

nr_servers = 15
avg_expt_time = 45  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for lr, gam, rep, lay in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--gamma {gam} "
        f"--num_layers {lay}"
    )
    print(expt_call, file=output_file)

output_file.close()
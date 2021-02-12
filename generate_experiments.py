#!/usr/bin/env python3
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/sloss'
base_call = (f"python train.py --dataset cifar10 "
             f"--dataset_path {DATA_HOME}/data "
             f"--layers 28 --widen-factor 2 "
             f"--epochs 200 "
             f"--print-freq 200 "
             f"--batch-size 250 "
             f"--checkpoint_dir {DATA_HOME}/logs/ ")

repeats = 1

learning_rate = [.1, .075, .05, 0.25]
sloss = [True]
lower_lim = [-5, -10, -15, -20, -50]

settings = [(lr, sloss_, l_lim, rep)
            for lr in learning_rate
            for sloss_ in sloss
            for l_lim in lower_lim
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, sloss_, l_lim, rep) in settings:
    expt_call = (
        f"{base_call} " +
        f"--lr {lr} " +
        f"--lower-limit {l_lim} " +
        (f"--no-sloss " if not sloss_ else "")
    )
    print(expt_call, file=output_file)

output_file.close()

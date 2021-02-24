#!/usr/bin/env python3
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/sloss'
base_call = (f"python train.py --dataset cifar100 "
             f"--dataset_path {DATA_HOME}/data "
             f"--layers 28 --widen-factor 10 "
             f"--epochs 200 "
             f"--print-freq 200 "
             f"--batch-size 250 "
             f"--checkpoint_dir {DATA_HOME}/logs/ ")

repeats = 1

learning_rate = [0.1, .075]
sloss = [True]
lower_lim = [-7, -7.5, -8, -8.5, -9]
upper_lim = [-3.5, -4, -4.5]
superclass = [False]

# superclass = [True, False]
# sloss = [False]
# lower_lim = [0]
# upper_lim = [0]

settings = [(lr, sloss_, l_lim, u_lim, sclass, rep)
            for lr in learning_rate
            for sloss_ in sloss
            for l_lim in lower_lim
            for u_lim in upper_lim
            for sclass in superclass
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, sloss_, l_lim, u_lim, sclass, rep) in settings:
    expt_call = (
        f"{base_call} " +
        f"--lr {lr} " +
        f"--lower-limit {l_lim} " +
        f"--upper-limit {u_lim} " +
        (f"--superclass " if sclass else "") +
        (f"--no-sloss " if not sloss_ else "")
    )
    print(expt_call, file=output_file)

output_file.close()


# def get_experiment_params(dataset, classes):
#     return {"num_classes": len(get_logic_terms(dataset, classes))}


# def get_class_ixs(dataset, classes):
#     return [t.ixs1 for t in get_logic_terms(dataset, classes)]
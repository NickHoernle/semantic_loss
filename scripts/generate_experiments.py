#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/vaelib'
base_call = (f"semi_supervised_vae.py gmm --input-data={DATA_HOME}/data --output-data={DATA_HOME}/output "
             f"--use_cuda=True "
             f"--num_epochs=300 "
             f"--num_labeled_data_per_class=400 "
             f"--s_loss=True"
             f"--num_test_samples=0 "
             f"--num_loader_workers=2 "
             f"--dataset=CIFAR10 ")

repeats = 1
learning_rates = [1e-2, 1e-4, 1e-6]
# learning_rates = [1e-3]
learning_rates2 = [1e-2, 1e-4, 1e-6] # these clearly need to be small
gammas = [.99]
hidden_dim = [500, 1000]
# hidden_dim = [100, 500]
kernel_nums= [200]
batch_size = [100]
# backward = [True, False]
# back_strength = [1e2, 1e3, 1e4, 1e5]
# num_layers = [40]
# gammas = [.99]

settings = [(lr, lr2, gam, h_dim, k_num, bs, rep)
            for lr in learning_rates
            for lr2 in learning_rates2
            for gam in gammas
            for h_dim in hidden_dim
            for k_num in kernel_nums
            for bs in batch_size
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 10
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, lr2, gam, h_dim, kn, bs, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr={lr} "
        f"--lr2={lr2} "
        f"--gamma={gam} "
        f"--hidden_dim={h_dim} "
        f"--batch_size={bs} "
        f"--kernel_num={kn} "
        f"run"
    )
    print(expt_call, file=output_file)

output_file.close()

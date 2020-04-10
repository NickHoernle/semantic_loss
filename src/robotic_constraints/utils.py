#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import shutil

# make nflib available
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import rss_code_and_data.code.curve_funcs as cf
from rss_code_and_data.code.dmp import (DMP,gen_weights,imitate_path,plot_rollout)


def generate_trajectories(n_data=500):

    constraints = [
        {"coords": [.25, .25], 'radius': .1},
        {"coords": [.5, .5], 'radius': .1},
        {"coords": [.8, .25], 'radius': .1},
        {"coords": [.7, .75], 'radius': .1}
    ]

    trajectories = []
    params = []
    weights = []

    while len(trajectories) < n_data:
        start_params, trajectory = cf.simpleCurveWithAvoidPoint(
            start_range=([0.0, 0.0], [0.1, 1.0]),
            goal_range=([0.9, 0.0], [1.0, 1.0]),
            #         start_range=([0, 0.5], [0.01,.51]),
            #         goal_range=([.99, 0.5], [1., 0.51]),
            attractor_range=([0.25, 0.25], [0.8, 0.8])
        )

        valid = True
        for constraint in constraints:
            if np.sum((trajectory[:, 0] - constraint['coords'][0]) ** 2 +
                      (trajectory[:, 1] - constraint['coords'][1]) ** 2 < constraint['radius'] ** 2) > 0:
                valid = False
                break

        if len(trajectory.reshape(-1,)) < 200:
            valid = False

        if valid:

            # now get the associated dmp weights
            dmp = DMP(25, dt=1 / 100, d=2)
            dmp.T = 100
            dmp.start = trajectory[0]
            dmp.y0 = trajectory[0]
            dmp.goal = trajectory[-1]

            path, dmp_weights = imitate_path(trajectory, dmp)

            trajectories.append(trajectory)
            weights.append(dmp_weights)

            params.append((start_params[0], start_params[-1]))

    return trajectories, weights, params, constraints


def plot_trajectories(trajectories, params=[], constraints=[], ax=None):

    for trajectory in trajectories:
        plt.plot(*trajectory.T, alpha=.1, c="C0")
        plt.scatter(*trajectory[0], c='C2', alpha=.1)
        plt.scatter(*trajectory[-1], c='C3', alpha=.1)

    ax = plt.gca()
    for constraint in constraints:
        circle = plt.Circle(constraint['coords'], constraint['radius'], color='b', fill=False)
        ax = plt.gca()
        ax.add_artist(circle)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    return ax


def save_trajectories(trajectories, weights, out_path, constraints=[]):

    for i, (trajectory, weights) in enumerate(zip(trajectories, weights)):
        file_path = os.path.join(out_path, f"trajectory_{i+1}.pt")
        torch.save(torch.tensor(trajectory).float(), file_path)

        file_path = os.path.join(out_path, f"weights_{i+1}.pt")
        torch.save(torch.tensor(weights).float(), file_path)

    all_ids = np.arange(1, len(trajectories)+1)
    size_data = len(all_ids)
    train = np.random.choice(all_ids, size=int(np.floor(0.7 * size_data)), replace=False)
    all_ids = all_ids[~np.in1d(all_ids, train)]
    validate = np.random.choice(all_ids, size=int(np.floor(0.15 * size_data)), replace=False)
    all_ids = all_ids[~np.in1d(all_ids, validate)]
    test = np.random.choice(all_ids, size=int(np.floor(0.15 * size_data)), replace=False)

    file_path = os.path.join(out_path, f"data_assignments.json")
    with open(file_path, 'w') as f:
        obj = {}
        obj['train'] = [int(u) for u in train]
        obj['test'] = [int(u) for u in test]
        obj['validate'] = [int(u) for u in validate]
        obj['constraints'] = constraints
        json.dump(obj, f)


def update_params_into_trajectories(out_path):
    file_path = os.path.join(out_path, f"data_assignments.json")
    with open(file_path, 'r') as f:
        assignments = json.load(f)

    assignments['params'] = {}

    for dset in ['train', 'validate', 'test']:
        ids = assignments[dset]
        for ID in ids:
            trajectory_path = os.path.join(out_path, f"trajectory_{ID}.pt")
            trajectory_dat = torch.load(trajectory_path)
            condition_params = [float(trajectory_dat[0, 0]),
                                float(trajectory_dat[0, 1]),
                                float(trajectory_dat[-1, 0]),
                                float(trajectory_dat[-1, 1])]
            assignments['params'][ID] = condition_params

    with open(file_path, 'w') as f:
        json.dump(assignments, f)


def create_and_save_data(output_dir, n_data=1000):

    # refresh the data
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    trajectories, weights, params, constraints = generate_trajectories(n_data)
    save_trajectories(trajectories, weights, output_dir, constraints)


if __name__ == "__main__":
    update_params_into_trajectories('../../data/robotic_constraints')
    # create_and_save_data('../../data/robotic_constraints/', 40000)


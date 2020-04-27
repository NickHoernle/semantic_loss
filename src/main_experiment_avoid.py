#!/usr/bin/env python

import argparse

from robotic_constraints.main_experiment_avoid import main
from robotic_constraints.main_experiment_avoid import construct_parser

# from robotic_constraints.main_vae_experiment_avoid import main
# from robotic_constraints.main_vae_experiment_avoid import construct_parser


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)

# python main_experiment_avoid.py -i ../data/robotic_constraints -o ../experiments/robotic_constraints --use_cuda False --batch_size 250 --num_epochs 500 --lr 1e-3 --backward True --hidden_dim 50 --gamma 0.9999 --max_grad_norm 1 --back_strength 1e2
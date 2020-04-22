#!/usr/bin/env python

import argparse

# from robotic_constraints.main_experiment_avoid import main
# from robotic_constraints.main_experiment_avoid import construct_parser

from robotic_constraints.main_vae_experiment_avoid import main
from robotic_constraints.main_vae_experiment_avoid import construct_parser


if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
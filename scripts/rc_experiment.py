#!/usr/bin/env python
import fire

from robotic_constraints.robotic_constraints_flow import RC_Flow

if __name__ == '__main__':
    fire.Fire(RC_Flow)

# robotic_constraints.py --input-data=data/robotic_constraints --output-data=experiments/robotic_constraints_vae --use_cuda=False run
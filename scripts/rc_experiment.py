#!/usr/bin/env python
import fire

from robotic_constraints.robotic_constraints_flow import RC_Flow

if __name__ == '__main__':
    fire.Fire(RC_Flow)

# rc_experiment.py --input-data=data/robotic_constraints --output-data=experiments/robotic_constraints_vae --lr=1e-3 --use_cuda=False --backward=True --batch_size=50 run
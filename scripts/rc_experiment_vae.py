#!/usr/bin/env python
import fire

from robotic_constraints.robotic_constraints_vae import RC_VAE

if __name__ == '__main__':
    fire.Fire(RC_VAE)

# rc_experiment_vae.py --input-data=data/robotic_constraints --output-data=experiments/robotic_constraints_vae --lr=1e-3 --use_cuda=False --backward=False --batch_size=50 run

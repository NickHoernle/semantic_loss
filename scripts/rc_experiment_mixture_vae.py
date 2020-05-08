#!/usr/bin/env python
import fire

from robotic_constraints.robotic_constraints_mixture_vae import RC_mixture_VAE

if __name__ == '__main__':
    fire.Fire(RC_mixture_VAE)

# rc_experiment_mixture_vae.py --input-data=data/robotic_constraints --output-data=experiments/robotic_constraints_vae --lr=1e-3 --use_cuda=False --backward=False --batch_size=50 run

#!/usr/bin/env python
import fire

from semi_supervised.semi_supervised_vae import VAESemiSupervisedTrainer

if __name__ == '__main__':
    fire.Fire(VAESemiSupervisedTrainer)

# semi_supervised_vae.py --input-data=data/vaelib/MNIST --output-data=experiments/semi_supervised --use_cuda=False
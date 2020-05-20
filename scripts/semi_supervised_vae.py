#!/usr/bin/env python
import fire

from semi_supervised.semi_supervised_vae import VAESemiSupervisedTrainer
from semi_supervised.semi_supervised_m2 import M2SemiSupervisedTrainer

if __name__ == '__main__':
    fire.Fire({
        "nick": VAESemiSupervisedTrainer,
        "m2": M2SemiSupervisedTrainer,
    })

# semi_supervised_vae.py --input-data=data/vaelib/MNIST --output-data=experiments/semi_supervised --use_cuda=False
# semi_supervised_vae.py --input-data=data/vaelib/MNIST --output-data=experiments/semi_supervised_vae --use_cuda=False --num_epochs=100 --hidden_dim=20 --batch_size=100 --lr=1e-4 --num_labeled_data_per_class=100 run
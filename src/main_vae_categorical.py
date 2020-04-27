#!/usr/bin/env python

from vaelib.pytorch_categorical_vae import main
from vaelib.pytorch_categorical_vae import construct_parser

if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)

# python main_vae_categorical.py -i ../data/vaelib -o ../experiments/vaelib --use_cuda False --batch_size 500 --num_epochs 500 --lr 1e-3 --hidden_dim 20 --gamma 0.99 --max_grad_norm 1 --dataset MNIST
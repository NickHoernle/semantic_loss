# Installation:
```
pip install -e .
```

# Execution:
Run CIFAR10 experiment:
```
run_image_experiments.py cifar10 --layers=10 --widen_factor=1
```

Run CIFAR100 experiment:
```
run_image_experiments.py cifar100 --layers=10 --widen_factor=1
```

# Help functions:
```
run_image_experiments.py -- --help
run_image_experiments.py cifar10 -- --help
run_image_experiments.py cifar100 -- --help
```

# Acknowledgements
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- [Wide Residual Networks (WideResNets) in PyTorch](https://github.com/xternalz/WideResNet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.

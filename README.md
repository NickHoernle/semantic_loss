# semantic_wmi_loss
Repository for the semantic WMI loss

A lot of the flows code has been adapted from: https://github.com/karpathy/pytorch-normalizing-flows

Houdini data folder : https://github.com/capergroup/houdini/tree/master/Data

```
conda create -n sloss python=3
conda activate sloss
pip install -e .
```

```
semi_supervised_vae.py --input-data=data/vaelib/MNIST --output-data=experiments/semi_supervised_vae --use_cuda=False --num_epochs=50 --hidden_dim=20 --batch_size=256 --lr=1e-3 --num_labeled_data_per_class=100 run
```

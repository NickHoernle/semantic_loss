#!/usr/bin/env python
import fire

from semi_supervised.semi_supervised_flow import FlowSemiSupervisedTrainer

if __name__ == '__main__':
    fire.Fire(FlowSemiSupervisedTrainer)

# semi_supervised_flows.py --input-data=data/vaelib/MNIST --output-data=experiments/semi_supervised_flow --use_cuda=False run
import os
import pydoc
import shutil
import numpy as np
import torch
from functools import wraps
import inspect


def save_checkpoint(state, is_best, experiment):
    """Saves checkpoint to disk"""
    if not os.path.exists(experiment.checkpoint_directory):
        os.makedirs(experiment.checkpoint_directory)
    torch.save(state, experiment.checkpoint)
    if is_best:
        shutil.copyfile(experiment.checkpoint, experiment.best_checkpoint)


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """

    @wraps(func)
    def wrapper(self, *args, **kargs):
        (
            names,
            varargs,
            varkw,
            defaults,
            kwonlyargs,
            kwonlydefaults,
            annotations,
        ) = inspect.getfullargspec(func)

        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def inherit_doc_for_fire(func):

    return lambda a, b: (a, b)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return np.mean(self.vals) * 100

    @property
    def sum(self):
        return np.sum(self.vals)

    @property
    def count(self):
        return len(self.vals)

    @property
    def val(self):
        return np.mean(self.vals[-self.n :]) * 100

    def reset(self):
        self.vals = []
        self.n = 100

    def update(self, vals, n=1):
        self.vals += list(vals)
        self.n = n

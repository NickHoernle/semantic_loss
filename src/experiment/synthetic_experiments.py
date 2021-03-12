import os
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from symbolic import symbolic
from symbolic import train
from symbolic.utils import (AccuracyMeter, AverageMeter, save_figure)
from experiment.datasets import (get_synthetic_loaders)
from experiment.generative import LinearVAE, ConstrainedVAE


class BaseSyntheticExperiment(train.Experiment):
    """Experimental setup for training with domain knowledge specified by a DNF logic formula on the synthetic dataset with continuous constraints. Wraps: train.Experiment.

    Synthetic Experiment Parameters:
        nhidden     num hidden units in the encoder / decoder respectively
        ndims       how many dimensions are we modeling
        name        name to use in logging the results
    \n
    """

    __doc__ += train.Experiment.__doc__

    def __init__(
        self,
        nhidden: int = 250,
        nlatent: int = 25,
        ndims: int = 2,
        baseline: bool = False,
        name: str = "Synthetic",
        size_of_train_set: int = 5000,
        **kwargs,
    ):
        self.nhidden = nhidden
        self.ndims = ndims
        self.nlatent = nlatent
        self.baseline = baseline
        self.name = name
        self.weight = 10.0
        self.size_of_train_set = size_of_train_set
        super().__init__(**kwargs)

    @property
    def params(self):
        return f"{self.name}-{self.lr}_{self.seed}_{self.baseline}"

    def pre_train_hook(self, train_loader):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
        train_loader.dataset.sampler.plot(ax=ax, with_term_labels=True)
        fig_file = os.path.join(self.figures_directory, f"generated_data.png")
        save_figure(fig, fig_file, self)

    def plot_validation_reconstructions(self, epoch, model, loader):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
        recons = []
        for i, data in enumerate(loader):
            model_input = self.get_input_data(data)
            with torch.no_grad():
                output = model(model_input, test=True)
                recon, (m, lv) = output
            recons += [recon]

        recons = torch.cat(recons, dim=0)
        valid_constraints = [t.valid(recons) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        ax.scatter(*recons[v_c].numpy().T, s=0.5, label="valid", c="C2")
        ax.scatter(*recons[~v_c].numpy().T, s=0.5, label="invalid", c="C3")
        ax.legend(loc="best")
        fig_file = os.path.join(self.figures_directory, f"{epoch}_reconstruction.png")
        save_figure(fig, fig_file, self)

    def epoch_finished_hook(self, *args, **kwargs):
        if not args[0] % 10 == 0:
            return
        self.plot_validation_reconstructions(*args)

    def get_loaders(self):
        train_size = self.size_of_train_set
        valid_size = 1000
        test_size = 5000

        return get_synthetic_loaders(train_size, valid_size, test_size)

    def create_model(self):
        if self.baseline:
            return LinearVAE(
                terms=self.logic_terms,
                ndims=self.ndims,
                nhidden=self.nhidden,
                nlatent=self.nlatent,
            )
        return ConstrainedVAE(
            terms=self.logic_terms,
            ndims=self.ndims,
            nhidden=self.nhidden,
            nlatent=self.nlatent,
        )

    def get_optimizer_and_scheduler(self, model, train_loader):
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(train_loader) * self.epochs
        )
        return optimizer, scheduler

    def init_meters(self):
        loss = AverageMeter()
        constraint = AccuracyMeter()
        self.losses = {
            "loss": loss,
            "constraint": constraint,
        }

    def get_input_data(self, data):
        samples = data[0].to(self.device)
        # labels = data[1].to(self.device)
        return samples

    def get_target_data(self, data):
        samples = data[0].to(self.device)
        labels = data[1].to(self.device)
        return samples

    def criterion(self, output, target, train=True):

        if not self.baseline and train:
            (recon, log_py), (mu, lv) = output
            ll = []
            for j, p in enumerate(recon.split(1, dim=1)):
                ll += [
                    self.weight
                    * F.mse_loss(p.squeeze(1), target, reduction="none").sum(dim=1)
                ]
            pred_loss = torch.stack(ll, dim=1)
            recon_losses, labels = pred_loss.min(dim=1)

            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
            loss = (log_py.exp() * (pred_loss + log_py)).sum(dim=1).mean()
            loss += recon_losses.mean()
            loss += F.nll_loss(log_py, labels)
            loss += kld

            return loss

        recon, (mu, lv) = output
        loss = (
            self.weight
            * F.mse_loss(recon.squeeze(1), target, reduction="none").sum(dim=1).mean()
        )
        loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        return loss

    def update_train_meters(self, loss, output, target):
        if not self.baseline:
            (recon, log_py), (mu, lv) = output
            preds = recon[np.arange(len(log_py)), log_py.argmax(dim=1)]
        else:
            preds, (mu, lv) = output

        self.losses["loss"].update(loss.data.item(), target.size(0))
        valid_constraints = [t.valid(preds) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        self.losses["constraint"].update(v_c.tolist(), v_c.size(0))

    def update_test_meters(self, loss, output, target):
        preds, (mu, lv) = output
        self.losses["loss"].update(loss.data.item(), target.size(0))
        valid_constraints = [t.valid(preds) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        self.losses["constraint"].update(v_c.tolist(), v_c.size(0))

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f'Loss {self.losses["loss"].val:.4f} ({self.losses["loss"].avg:.4f})\t'
            f'Constraint {self.losses["constraint"].val:.4f} ({self.losses["constraint"].avg:.4f})\n'
        )

    def iter_done(self, type="Train"):
        text = (
            f'{type}: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Constraint {round(self.losses["constraint"].avg, 3)}\n'
        )
        print(text, end="")
        self.logfile.write(text + "\n")

    def update_best(self, val):
        if val < self.best_loss:
            self.best_loss = val
            return True
        return False


class FullyKnownConstraintsSyntheticExperiment(BaseSyntheticExperiment):
    def __init__(self, name: str = "SyntheticFull", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def logic_terms(self):
        ll = 0.5
        return [
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=np.pi,
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=np.pi / 4,
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=2 * np.pi / 4,
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=3 * np.pi / 4,
            ),
            symbolic.Box(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=5 * np.pi / 4,
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=6 * np.pi / 4,
            ),
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=7 * np.pi / 4,
            ),
        ]


class PartiallyKnownConstraintsSyntheticExperiment(
    FullyKnownConstraintsSyntheticExperiment
):
    def __init__(self, name: str = "SyntheticPartial", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_loaders(self):
        train_size = self.size_of_train_set
        valid_size = 1000
        test_size = 5000
        sampler_params = {
            "rotations": [
                # 0,
                np.pi / 4,
                2 * np.pi / 4,
                3 * np.pi / 4,
                # np.pi,
                5 * np.pi / 4,
                6 * np.pi / 4,
                7 * np.pi / 4,
            ]
        }
        return get_synthetic_loaders(
            train_size, valid_size, test_size, sampler_params=sampler_params
        )


synthetic_experiment_options = {
    "synthetic_full": FullyKnownConstraintsSyntheticExperiment,
    "synthetic_partial": PartiallyKnownConstraintsSyntheticExperiment,
}

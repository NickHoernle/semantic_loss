from experiment.generative import LinearVAE, ConstrainedVAE
from experiment.datasets import (get_synthetic_loaders)
from symbolic.utils import (AccuracyMeter, AverageMeter, save_figure)
from symbolic import train
from symbolic import symbolic
from training.supervised.oracles import DL2_Oracle
import dl2lib.query as q
import dl2lib as dl2
import pdb
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


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
        return f"{self.name}-{self.lr}_{self.seed}_{self.baseline}_{self.size_of_train_set}"

    def pre_train_hook(self, train_loader):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
        train_loader.dataset.sampler.plot(ax=ax, with_term_labels=False)
        fig_file = os.path.join(self.figures_directory, f"generated_data.png")

        ax.legend(loc="best")
        # draw the constraints here
        for term in self.logic_terms:

            p1 = term.rotate(term.lims[[0, 1], [0, 1]][None, :]).squeeze()
            p2 = term.rotate(term.lims[[0, 1], [0, 0]][None, :]).squeeze()
            p3 = term.rotate(term.lims[[0, 1], [1, 1]][None, :]).squeeze()
            p4 = term.rotate(term.lims[[0, 1], [1, 0]][None, :]).squeeze()

            # import pdb
            # pdb.set_trace()

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c="C3")
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], c="C3")
            ax.plot([p2[0], p4[0]], [p2[1], p4[1]], c="C3")
            ax.plot([p1[0], p3[0]], [p1[1], p3[1]], c="C3")

        save_figure(fig, fig_file, self)

    def plot_validation_reconstructions(self, epoch, model, loader):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
        recons = []
        for i, data in enumerate(loader):
            model_input = self.get_input_data(data)
            with torch.no_grad():
                output = model(model_input, test=True)
                recon, (m, lv), _ = output
            recons += [recon]

        recons = torch.cat(recons, dim=0)
        valid_constraints = [t.valid(recons) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        ax.scatter(*recons[v_c].numpy().T, s=0.5, label="valid", c="C2")
        ax.scatter(*recons[~v_c].numpy().T, s=0.5, label="invalid", c="C3")
        ax.legend(loc="best")
        fig_file = os.path.join(self.figures_directory,
                                f"{epoch}_reconstruction.png")
        save_figure(fig, fig_file, self)

    def plot_prior_samples(self, epoch, model, loader):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()

        z = torch.randn(10000, self.nlatent)
        recons = model.decode(z).detach()

        valid_constraints = [t.valid(recons) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        ax.scatter(*recons[v_c].numpy().T, s=0.5, label="valid", c="C2")
        ax.scatter(*recons[~v_c].numpy().T, s=0.5, label="invalid", c="C3")
        ax.legend(loc="best")
        fig_file = os.path.join(self.figures_directory,
                                f"{epoch}_prior_samples.png")
        save_figure(fig, fig_file, self)

    def epoch_finished_hook(self, *args, **kwargs):
        if not args[0] % 10 == 0:
            return
        self.plot_validation_reconstructions(*args)
        self.plot_prior_samples(*args)

    def get_loaders(self):
        train_size = self.size_of_train_set
        valid_size = 1000
        test_size = 5000

        return get_synthetic_loaders(train_size, valid_size, test_size)

    def create_model(self):
        print('called')
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

    def criterion(self, output, target):

        if not self.baseline:
            (recon, log_py), (mu, lv), log_prior = output
            ll = []
            for j, p in enumerate(recon.split(1, dim=1)):
                ll += [
                    self.weight
                    * F.mse_loss(p.squeeze(1), target, reduction="none").sum(dim=1)
                ]
            pred_loss = torch.stack(ll, dim=1)
            recon_losses, labels = pred_loss.min(dim=1)

            kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
            loss = (log_py.exp() * (pred_loss + log_py - log_prior)
                    ).sum(dim=1).mean()
            loss += recon_losses.mean()
            loss += F.nll_loss(log_py, labels)
            loss += kld

            return loss

        recon, (mu, lv), _ = output
        loss = (
            self.weight
            * F.mse_loss(recon.squeeze(1), target, reduction="none").sum(dim=1).mean()
        )
        loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        return loss

    def update_train_meters(self, loss, output, target):
        if not self.baseline:
            (recon, log_py), (mu, lv), log_prior = output
            preds = recon[np.arange(len(log_py)), log_py.argmax(dim=1)]
        else:
            preds, (mu, lv), _ = output

        self.losses["loss"].update(loss.data.item(), target.size(0))
        valid_constraints = [t.valid(preds) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        # print('vc', v_c)
        # print('list , sizes', v_c.tolist(), v_c.size(0), v_c.size())
        # print('constraint', self.losses["constraint"])
        self.losses["constraint"].update(v_c.tolist(), v_c.size(0))

    def update_test_meters(self, loss, output, target):
        if not self.baseline:
            (recon, log_py), (mu, lv), log_prior = output
            preds = recon[np.arange(len(log_py)), log_py.argmax(dim=1)]
        else:
            preds, (mu, lv), _ = output

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

    def iter_done(self, epoch, type="Train"):
        text = (
            f'{type} [{epoch+1}/{self.epochs}]: Loss {round(self.losses["loss"].avg, 3)}\t '
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
            symbolic.RotatedBox(
                constrained_ixs=[0, 1],
                not_constrained_ixs=[],
                lims=((-ll, ll), (-5.5, -2.5)),
                theta=0,
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


class DL2SyntheticExperiment(PartiallyKnownConstraintsSyntheticExperiment):
    """[summary]

    Args:
        BaseSyntheticExperiment ([type]): [description]
    """

    def __init__(self, name: str = "SyntheticDL2", **kwargs):
        """[summary]

        Args:
            name (str, optional): [description]. Defaults to "SyntheticDL2".
        """
        # Setting baseline to True results in creating the default LinearVAE.
        kwargs['baseline'] = True
        super().__init__(name=name, **kwargs)
        print('before')
        self.model = self.create_model()
        # self.oracle = DL2_Oracle(learning_rate=0.01, net=self.model, constraint=constraint, use_cuda=use_cuda)
        parser = argparse.ArgumentParser(description='desc')
        # parser.add("--eps-const", type=float, default=1e-5, required=False, help="the epsilon for boolean constants")
        parser.add("--eps-check", type=float, default=0, required=False,
                   help="the epsilon for checking comparisons of floating point values; note that a nonzero value slightly changes the semantics of DL2")
        parser.add_argument('--or', default='mul', type=str,
                            help='help this is a hack')
        try:
            self._args = parser.parse_args()
            self._args = parser.parse_known_args()[0]
        except:
            self._args = parser.parse_known_args()[0]
        self.dl2_multiplier = 1e-4
        print("done")

    def __rotate_around_origin(self, xy, radians):
        """[summary]

        Args:
            xy ([type]): [description]
            radians ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Only rotate a point around the origin (0, 0).
        x, y = xy
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)

        return xx, yy

    def __box_to_constraint(self, box, point):
        # x, y = point
        box_conditions = []

        x_coords = [x for x, y in box]
        y_coords = [y for x, y in box]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        conditions = dl2.And([
            dl2.GEQ(point[:, 0], min_x),
            dl2.LEQ(point[:, 0], max_x),
            dl2.GEQ(point[:, 1], min_y),
            dl2.LEQ(point[:, 1], max_y)
        ])
        return conditions

    def criterion(self, output, target):
        """[summary]

        Args:
            output ([type]): [description]
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        recon, (mu, lv), _ = output

        thetas = [
            np.pi,
            np.pi / 4,
            2 * np.pi / 4,
            3 * np.pi / 4,
            0,
            5 * np.pi / 4,
            6 * np.pi / 4,
            7 * np.pi / 4
        ]
        ll = 0.5
        box = [(-ll, -5.5), (ll, -5.5), (ll, -2.5), (-ll, -2.5)]
        constraints = []
        for theta in thetas:
            rotation_matrix = torch.tensor(
                [
                    [np.cos(-theta), -np.sin(-theta)],
                    [np.sin(-theta), np.cos(-theta)],
                ]
            ).float()
            rotated_recon = recon.mm(rotation_matrix)
            constraints.append(self.__box_to_constraint(box, rotated_recon))
        constraint = dl2.Or(constraints)

        # pdb.set_trace()
        # self._satisfied = constraint.satisfy(self._args)

        # reconstruction loss
        loss = self.weight * \
            F.mse_loss(recon, target, reduction="none").sum(dim=1)

        # dl2 constraint loss
        loss += self.dl2_multiplier * (constraint.loss(self._args))

        # KLD loss
        loss += -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        return loss.mean()


synthetic_experiment_options = {
    "synthetic_full": FullyKnownConstraintsSyntheticExperiment,
    "synthetic_partial": PartiallyKnownConstraintsSyntheticExperiment,
    "synthetic_dl2": DL2SyntheticExperiment
}

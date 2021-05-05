import os

import torch
import torch.nn.functional as F
import numpy as np

from symbolic import train
from symbolic.symbolic import ConstantEqualityGenerative
from symbolic.utils import AccuracyMeter, AverageMeter, save_figure
from experiment.datasets import get_train_valid_loader, get_test_loader, resampled_train
from experiment.generative import MnistVAE, ConstrainedMnistVAE
from torch.distributions.normal import Normal
from experiment.class_mapping import mnist_domain_knowledge as knowledge
from experiment.class_mapping import flat_class_mapping as flat_knowledge

from torch.distributions.categorical import Categorical

import matplotlib
import matplotlib.pyplot as plt
import random
import math

matplotlib.use("Agg")

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


class BaseMNISTExperiment(train.Experiment):
    """
    Experimental setup for training with domain knowledge specified by a DNF logic formula on the synthetic dataset
    with continuous constraints. Wraps: train.Experiment.

    """

    __doc__ += train.Experiment.__doc__

    def __init__(
        self,
        sloss: bool = False,
        name: str = "MNIST",
        hidden_dim1: int = 250,
        hidden_dim2: int = 100,
        zdim: int = 10,
        **kwargs,
    ):
        self.dataset = "mnist"
        self.sloss = sloss
        self.name = name
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.zdim = zdim
        super().__init__(**kwargs)

    @property
    def params(self):
        return f"{self.name}-{self.lr}_{self.seed}_{self.sloss}_{self.batch_size}"

    def get_loaders(self):
        train_loader, val_loader, classes, train_indexes = get_train_valid_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            augment=False,
            random_seed=self.seed,
            valid_size=0.3,
            shuffle=True,
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            do_normalize=False,
        )

        test_loader = get_test_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            dataset=self.dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            do_normalize=False,
        )

        self.train_indexes = train_indexes
        self.classes = classes

        return train_loader, val_loader, test_loader

    def train_loader_shuffler(self, train_loader):
        return resampled_train(
            train_idx=self.train_indexes,
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            augment=False,
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            do_normalize=False,
        )

    def create_model(self):
        return MnistVAE(
            x_dim=784,
            h_dim1=self.hidden_dim1,
            h_dim2=self.hidden_dim2,
            z_dim=self.zdim,
            num_labels=10,
        )

    def pre_train_hook(self, loader):
        pass

    def epoch_finished_hook(self, epoch, model, val_loader):
        pass

    def plot_model_samples(self, epoch, model):
        fig, axes = plt.subplots(10, 10, figsize=(20, 15))
        model.eval()

        for i, row in enumerate(axes):
            z_ = torch.randn((1, self.zdim))
            for j, ax in enumerate(row):
                # y_onehot = torch.zeros((1, 10)).float()
                # y_onehot[:, j] = 1
                # lbl = torch.ones(1).long() * j
                # mu_p, lv_p = model.get_priors(lbl)
                # std = torch.exp(0.5 * lv_p)
                # z = mu_p + std * z_

                recon = model.decode_one(z_, label=j)

                ax.imshow(
                    (
                        torch.sigmoid(recon[0]).view(
                            28, 28).detach().numpy() * 255
                    ).astype(np.uint8),
                    cmap="gray_r",
                )
                ax.grid(False)
                ax.set_axis_off()
        fig_file = os.path.join(self.figures_directory,
                                f"sample_epoch_{epoch}.png")
        save_figure(fig, fig_file, self)

    def plot_sampled_images(self, loader):
        fig, axes = plt.subplots(15, 3, figsize=(10, 45))
        for i, data in enumerate(loader):
            for j in range(15):
                if j >= 15:
                    break
                (
                    (in_data1, in_target1),
                    (in_data2, in_target2),
                    (in_data3, in_target3),
                ) = data
                axes[j, 0].imshow(
                    (in_data1[j, 0].numpy() * 255).astype(np.uint8), cmap="gray_r"
                )
                axes[j, 0].set_title(in_target1[j])
                axes[j, 1].imshow(
                    (in_data2[j, 0].numpy() * 255).astype(np.uint8), cmap="gray_r"
                )
                axes[j, 1].set_title(in_target2[j])
                axes[j, 2].imshow(
                    (in_data3[j, 0].numpy() * 255).astype(np.uint8), cmap="gray_r"
                )
                axes[j, 2].set_title(in_target3[j])
                axes[j, 0].grid(False)
                axes[j, 0].set_axis_off()
                axes[j, 1].grid(False)
                axes[j, 1].set_axis_off()
                axes[j, 2].grid(False)
                axes[j, 2].set_axis_off()
            break

        fig_file = os.path.join(self.figures_directory, f"example_data.png")
        save_figure(fig, fig_file, self)

    def get_optimizer_and_scheduler(self, model, train_loader):
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(train_loader) * self.epochs
        )
        return optimizer, scheduler

    def init_meters(self):
        loss = AverageMeter()
        acc1 = AccuracyMeter()
        acc2 = AccuracyMeter()
        acc3 = AccuracyMeter()
        entropy = AverageMeter()
        self.losses = {
            "loss": loss,
            "accuracy_class_1": acc1,
            "accuracy_class_2": acc2,
            "accuracy_class_3": acc3,
            "entropy": entropy,
        }

    def get_input_data(self, data):
        (in_data1, in_target1), (in_data2, in_target2), (in_data3, in_target3) = data

        in_data1 = in_data1.to(self.device).view(len(in_data1), -1)
        in_data2 = in_data2.to(self.device).view(len(in_data2), -1)
        in_data3 = in_data3.to(self.device).view(len(in_data3), -1)

        return (in_data1, in_data2, in_data3)

    def get_target_data(self, data):
        (in_data1, in_target1), (in_data2, in_target2), (in_data3, in_target3) = data

        in_data1 = in_data1.to(self.device).view(len(in_data1), -1)
        in_data2 = in_data2.to(self.device).view(len(in_data2), -1)
        in_data3 = in_data3.to(self.device).view(len(in_data3), -1)

        in_target1 = in_target1.to(self.device)
        in_target2 = in_target2.to(self.device)
        in_target3 = in_target3.to(self.device)

        in_data = (in_data1, in_data2, in_data3)
        labels = (in_target1, in_target2, in_target3)

        return in_data, labels

    def criterion(self, output, target, train=True):

        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (recons1, recons2, recons3), (lp1, lp2, lp3), logpy = output
        ll1, ll2, ll3 = [], [], []

        for i in range(10):
            ll1.append(calc_ll(recons1[i], tgt1))

        for j in range(10):
            ll2.append(calc_ll(recons2[j], tgt2))

        for k in range(10):
            ll3.append(calc_ll(recons3[k], tgt3))

        return (
            (lp1.exp() * (torch.stack(ll1, dim=1) + lp1)).sum(dim=1).mean()
            + (lp2.exp() * (torch.stack(ll2, dim=1) + lp2)).sum(dim=1).mean()
            + (lp3.exp() * (torch.stack(ll3, dim=1) + lp3)).sum(dim=1).mean()
        )

    def update_train_meters(self, loss, output, target):
        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (recons1, recons2, recons3), (lp1, lp2, lp3), logpy = output

        # TODO: note you should actually measure the most likely prediction...
        self.losses["loss"].update(loss.data.item(), tgt1.size(0))
        self.losses["accuracy_class_1"].update(
            (lp1.argmax(dim=1) == lbl1).tolist(), tgt1.size(0)
        )
        self.losses["accuracy_class_2"].update(
            (lp2.argmax(dim=1) == lbl2).tolist(), tgt2.size(0)
        )
        self.losses["accuracy_class_3"].update(
            (lp3.argmax(dim=1) == lbl3).tolist(), tgt3.size(0)
        )
        self.losses["entropy"].update(
            (
                -(lp1.exp() * lp1).sum(dim=1)
                - (lp2.exp() * lp2).sum(dim=1)
                - (lp3.exp() * lp3).sum(dim=1)
            )
            .mean()
            .data.item(),
            tgt3.size(0),
        )

    def update_test_meters(self, loss, output, target):
        self.update_train_meters(loss, output, target)

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {round(batch_time.val, 3)} ({round(batch_time.avg, 3)})\t"
            f'Loss {round(self.losses["loss"].val, 3)} ({self.losses["loss"].avg})\t'
            f'Acc1 {round(self.losses["accuracy_class_1"].val, 3)} ({round(self.losses["accuracy_class_1"].avg, 3)})\t'
            f'Acc2 {round(self.losses["accuracy_class_2"].val, 3)} ({round(self.losses["accuracy_class_2"].avg, 3)})\t'
            f'Acc3 {round(self.losses["accuracy_class_3"].val, 3)} ({round(self.losses["accuracy_class_3"].avg, 3)})\t'
            f'Ent {round(self.losses["entropy"].val, 3)} ({round(self.losses["entropy"].avg, 3)})\n'
        )

    def iter_done(self, epoch, type="Train"):
        text = (
            f'[{epoch + 1}/{self.epochs}]: {type}: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc1 {round(self.losses["accuracy_class_1"].avg, 3)} \t'
            f'Acc2 {round(self.losses["accuracy_class_2"].avg, 3)} \t'
            f'Acc3 {round(self.losses["accuracy_class_3"].avg, 3)} \t'
            f'Ent {round(self.losses["entropy"].avg, 3)} \n'
        )

        print(text, end="")
        self.logfile.write(text + "\n")

    def update_best(self, val):
        if val < self.best_loss:
            self.best_loss = val
            return True
        return False


def calc_ll(params, target, beta=1.0):
    """
    Helper to calculate the ll of a single prediction for one of the images that are being processed
    """
    recon, mu, lv, mu_prior, lv_prior, z = params

    # std = torch.exp(0.5 * lv)
    # std_prior = torch.exp(0.5 * lv_prior)

    rcon = F.binary_cross_entropy_with_logits(recon, target, reduction="none").sum(
        dim=-1
    )

    kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)

    return rcon + kld


def select_action(device, n_actions, best_guess):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        return best_guess
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


class ConstrainedMNIST(BaseMNISTExperiment):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs["sloss"] = True
        self.beta = 1.0
        super().__init__(**kwargs)

    @property
    def logic_terms(self):
        terms = []
        for k, vals in knowledge.items():
            for v0, v1 in vals:
                constrain = [v0, v1, k]
                lwr_c = np.arange(10)
                lwr_c = [
                    lwr_c[~np.isin(lwr_c, constrain[0])],
                    lwr_c[~np.isin(lwr_c, constrain[1])],
                    lwr_c[~np.isin(lwr_c, constrain[2])],
                ]

                terms.append(
                    ConstantEqualityGenerative(
                        ixs_active=constrain,
                        ixs_inactive=lwr_c,
                    )
                )
        return terms

    def create_model(self):
        self.model = ConstrainedMnistVAE(
            x_dim=784,
            h_dim1=self.hidden_dim1,
            h_dim2=self.hidden_dim2,
            z_dim=self.zdim,
            num_labels=10,
            terms=self.logic_terms,
        )
        return self.model

    def criterion(self, output, target, train=True):

        weight = np.max([0, self.beta])

        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (r1, r2, r3), (lp1, lp2, lp3), logpy = output

        # reconstruction accuracies
        ll1 = torch.stack([calc_ll(r, tgt1) for r in r1], dim=1)
        ll2 = torch.stack([calc_ll(r, tgt2) for r in r2], dim=1)
        ll3 = torch.stack([calc_ll(r, tgt3) for r in r3], dim=1)

        llik, ll = self.model.logic_decoder(
            (ll1, ll2, ll3, lp1, lp2, lp3), logpy)

        recon, labels = llik.min(dim=1)
        loss = (1 - weight)*(logpy.exp() * (llik + logpy)).sum(dim=1).mean()
        loss += weight * recon.mean()

        return loss

    def iter_start_hook(self, iteration_count, epoch, model, data):
        if epoch < 5:
            model.encoder.eval()
            model.label_encoder_dec1.eval()
            model.mu.eval()
            model.lv.eval()

        if len(data[0][0]) <= 1:
            return False
        return True

    def init_meters(self):
        loss = AverageMeter()
        mae = AverageMeter()
        acc = AccuracyMeter()
        entropy = AverageMeter()
        self.losses = {
            "loss": loss,
            "accuracy": acc,
            "entropy": entropy,
            "mae": mae
        }

    def update_train_meters(self, loss, output, target):
        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (recons1, recons2, recons3), (lp1, lp2, lp3), logpy = output

        vals = torch.tensor(flat_knowledge).to(self.device)
        pred1 = vals[:, 0][logpy.argmax(dim=1)]
        pred2 = vals[:, 1][logpy.argmax(dim=1)]
        pred3 = vals[:, 2][logpy.argmax(dim=1)]

        acc1 = ((pred1 == lbl1).float()).tolist()
        acc2 = ((pred2 == lbl2).float()).tolist()
        acc3 = ((pred3 == lbl3).float()).tolist()

        self.losses["loss"].update(loss.data.item(), tgt1.size(0))
        self.losses["accuracy"].update(acc1, tgt3.size(0))
        self.losses["accuracy"].update(acc2, tgt3.size(0))
        self.losses["accuracy"].update(acc3, tgt3.size(0))
        self.losses["entropy"].update(
            (-(logpy.exp() * logpy).sum(dim=1)).mean().data.item(),
            tgt3.size(0),
        )
        self.losses["mae"].update(
            (
                (pred1 - lbl1).float().pow(2).sqrt() +
                (pred2 - lbl2).float().pow(2).sqrt() +
                (pred3 - lbl3).float().pow(2).sqrt()
            ).mean().data.item(),
            lbl3.size(0)
        )

    def epoch_finished_hook(self, epoch, model, val_loader):
        if self.device == "cpu":
            self.plot_model_samples(epoch, model)
        if epoch > 2:
            self.beta -= 0.05

    def update_test_meters(self, loss, output, target):
        self.update_train_meters(loss, output, target)

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {round(batch_time.val, 3)} ({round(batch_time.avg, 3)})\t"
            f'Loss {round(self.losses["loss"].val, 3)} ({self.losses["loss"].avg})\t'
            f'Acc {round(self.losses["accuracy"].val, 3)} ({round(self.losses["accuracy"].avg, 3)})\t'
            f'MAE {round(self.losses["mae"].val, 3)} ({round(self.losses["mae"].avg, 3)})\t'
            f'Ent {round(self.losses["entropy"].val, 3)} ({round(self.losses["entropy"].avg, 3)})\n'
        )

    def iter_done(self, epoch, type="Train"):
        text = (
            f'{type} [{epoch+1}/{self.epochs}]: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc {round(self.losses["accuracy"].avg, 3)} \t'
            f'MAE {round(self.losses["mae"].avg, 3)} \t'
            f'Ent {round(self.losses["entropy"].avg, 3)} \n'
        )

        print(text, end="")
        self.logfile.write(text + "\n")


mnist_experiment_options = {
    "mnist_base": BaseMNISTExperiment,
    "mnist_with_constraints": ConstrainedMNIST,
}

import torch
import torch.nn.functional as F
import numpy as np

from symbolic import train
from symbolic.symbolic import ConstantConstraint
from symbolic.utils import (AccuracyMeter, AverageMeter)
from experiment.datasets import (get_train_valid_loader, get_test_loader)
from experiment.generative import MnistVAE, ConstrainedMnistVAE
from torch.distributions.normal import Normal
from experiment.class_mapping import mnist_domain_knowledge as knowledge


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
        hidden_dim1: int = 500,
        hidden_dim2: int = 250,
        zdim: int = 25,
        **kwargs,
    ):
        self.dataset = "mnist"
        self.sloss = sloss
        self.name = name
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.zdim = zdim
        self.beta = 50.0
        super().__init__(**kwargs)

    @property
    def params(self):
        return f"{self.name}-{self.lr}_{self.seed}_{self.sloss}"

    def get_loaders(self):
        train_loader, val_loader, classes = get_train_valid_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            augment=False,
            random_seed=self.seed,
            valid_size=0.1,
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

        self.classes = classes

        return train_loader, val_loader, test_loader

    def create_model(self):
        return MnistVAE(
            x_dim=784,
            h_dim1=self.hidden_dim1,
            h_dim2=self.hidden_dim2,
            z_dim=self.zdim,
            num_labels=10,
        )

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
        in_data1 = in_data1.to(self.device)
        in_data2 = in_data2.to(self.device)
        in_data3 = in_data3.to(self.device)

        return (in_data1, in_data2, in_data3)

    def get_target_data(self, data):
        (in_data1, in_target1), (in_data2, in_target2), (in_data3, in_target3) = data
        in_data1 = in_data1.to(self.device).reshape(len(in_data1), -1)
        in_data2 = in_data2.to(self.device).reshape(len(in_data2), -1)
        in_data3 = in_data3.to(self.device).reshape(len(in_data3), -1)

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

    std = torch.exp(0.5 * lv)
    std_prior = torch.exp(0.5 * lv_prior)

    kld = (Normal(mu, std).log_prob(z) - Normal(mu_prior, std_prior).log_prob(z)).sum(
        dim=1
    )
    rcon = F.binary_cross_entropy_with_logits(recon, target, reduction="none").sum(
        dim=1
    )

    return rcon + beta * kld


class ConstrainedMNIST(BaseMNISTExperiment):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs["sloss"] = True
        super().__init__(**kwargs)

    @property
    def logic_terms(self):
        terms = []
        for k, vals in knowledge.items():
            for v in vals:
                constrain = [v[0], 10 + v[1], 20 + k]
                lwr_c = np.arange(30)
                lwr_c = lwr_c[~np.isin(lwr_c, constrain)].tolist()

                terms.append(
                    ConstantConstraint(
                        ixs1=constrain,
                        ixs_not=[],
                        ixs_less_than=lwr_c,
                        threshold_upper=1.0,
                        threshold_lower=-10.0,
                        threshold_limit=-10.0,
                    )
                )
        return terms

    def create_model(self):
        return ConstrainedMnistVAE(
            x_dim=784,
            h_dim1=self.hidden_dim1,
            h_dim2=self.hidden_dim2,
            z_dim=self.zdim,
            num_labels=10,
            terms=self.logic_terms,
        )

    def criterion(self, output, target, train=True):

        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (r1, r2, r3), (lp1, lp2, lp3), logpy = output

        # reconstruction accuracies
        ll1 = torch.stack([calc_ll(r, tgt1) for r in r1], dim=1).unsqueeze(1)
        ll2 = torch.stack([calc_ll(r, tgt2) for r in r2], dim=1).unsqueeze(1)
        ll3 = torch.stack([calc_ll(r, tgt3) for r in r3], dim=1).unsqueeze(1)

        lp1 = lp1.log_softmax(dim=-1)
        lp2 = lp2.log_softmax(dim=-1)
        lp3 = lp3.log_softmax(dim=-1)

        llik = (
            (lp1.exp() * (ll1 + lp1)).sum(dim=-1)
            + (lp2.exp() * (ll2 + lp2)).sum(dim=-1)
            + (lp3.exp() * (ll3 + lp3)).sum(dim=-1)
        )

        return (logpy.exp() * (llik + logpy)).sum(dim=-1).mean()

    def init_meters(self):
        loss = AverageMeter()
        acc = AccuracyMeter()
        entropy = AverageMeter()
        self.losses = {
            "loss": loss,
            "accuracy": acc,
            "entropy": entropy,
        }

    def update_train_meters(self, loss, output, target):
        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (recons1, recons2, recons3), (lp1, lp2, lp3), logpy = output

        vals = torch.tensor([k for k, vals in knowledge.items() for v in vals])[
            None, :
        ].repeat(len(logpy), 1).to(self.device)
        acc = (
            (vals[np.arange(len(logpy)), logpy.argmax(dim=1)] == lbl3).float()
        ).tolist()

        self.losses["loss"].update(loss.data.item(), tgt1.size(0))
        self.losses["accuracy"].update(acc, tgt3.size(0))
        self.losses["entropy"].update(
            (-(logpy.exp() * logpy).sum(dim=1)).mean().data.item(),
            tgt3.size(0),
        )

    def update_test_meters(self, loss, output, target):
        self.update_train_meters(loss, output, target)

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {round(batch_time.val, 3)} ({round(batch_time.avg, 3)})\t"
            f'Loss {round(self.losses["loss"].val, 3)} ({self.losses["loss"].avg})\t'
            f'Acc {round(self.losses["accuracy"].val, 3)} ({round(self.losses["accuracy"].avg, 3)})\t'
            f'Ent {round(self.losses["entropy"].val, 3)} ({round(self.losses["entropy"].avg, 3)})\n'
        )

    def iter_done(self, epoch, type="Train"):
        text = (
            f'{type}: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc {round(self.losses["accuracy"].avg, 3)} \t'
            f'Ent {round(self.losses["entropy"].avg, 3)} \n'
        )

        print(text, end="")
        self.logfile.write(text + "\n")


mnist_experiment_options = {
    "mnist_base": BaseMNISTExperiment,
    "mnist_with_constraints": ConstrainedMNIST,
}

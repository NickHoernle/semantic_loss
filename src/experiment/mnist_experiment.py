import os

import torch
import torch.nn.functional as F
import numpy as np

from symbolic import train
from symbolic.symbolic import ConstantConstraint, GEQConstant
from symbolic.utils import AccuracyMeter, AverageMeter, save_figure
from experiment.datasets import get_train_valid_loader, get_test_loader
from experiment.generative import MnistVAE, ConstrainedMnistVAE
from torch.distributions.normal import Normal
from experiment.class_mapping import mnist_domain_knowledge as knowledge

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


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

    def pre_train_hook(self, loader):
        pass
        # self.plot_sampled_images(loader)

    def epoch_finished_hook(self, epoch, model, val_loader):
        pass
        # if (epoch + 1) % 10 == 0:
        #     self.plot_model_samples(epoch, model)

    def plot_model_samples(self, epoch, model):
        fig, axes = plt.subplots(10, 10, figsize=(20, 15))

        for i, row in enumerate(axes):
            for j, ax in enumerate(row):
                y_onehot = torch.zeros((1, 10)).float()
                y_onehot[:, j] = 1

                mu_p, lv_p = model.get_priors(y_onehot)
                std = torch.exp(0.5 * lv_p)
                z = mu_p + std * torch.randn((1, 10))

                recon = model.decode_one(z)

                ax.imshow((torch.sigmoid(recon[0]).view(28, 28).detach().numpy() * 255).astype(np.uint8), cmap='gray_r')
                ax.grid(False)
                ax.set_axis_off()
        fig_file = os.path.join(self.figures_directory, f"sample_epoch_{epoch}.png")
        save_figure(fig, fig_file, self)

    def plot_sampled_images(self, loader):
        fig, axes = plt.subplots(15, 3, figsize=(10, 45))
        for i, data in enumerate(loader):
            for j in range(15):
                if j >= 15:
                    break
                (in_data1, in_target1), (in_data2, in_target2), (in_data3, in_target3) = data
                axes[j, 0].imshow((in_data1[j, 0].numpy() * 255).astype(np.uint8), cmap='gray_r')
                axes[j, 0].set_title(in_target1[j])
                axes[j, 1].imshow((in_data2[j, 0].numpy() * 255).astype(np.uint8), cmap='gray_r')
                axes[j, 1].set_title(in_target2[j])
                axes[j, 2].imshow((in_data3[j, 0].numpy() * 255).astype(np.uint8), cmap='gray_r')
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
        # optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=.9)
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

    std = torch.exp(0.5 * lv)
    std_prior = torch.exp(0.5 * lv_prior)

    kld = (Normal(mu, std).log_prob(z) - Normal(mu_prior, std_prior).log_prob(z)).sum(
        dim=1
    )
    rcon = F.binary_cross_entropy_with_logits(recon, target, reduction="none").sum(
        dim=-1
    )

    return rcon + beta * kld


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
                constrain = [v0, 10 + v1, 20 + k]
                lwr_c = np.arange(30)
                lwr_c = lwr_c[~np.isin(lwr_c, constrain)].tolist()

                terms.append(
                    ConstantConstraint(
                        ixs1=constrain,
                        ixs_not=[],
                        ixs_less_than=lwr_c,
                        threshold_upper=-1,
                        threshold_lower=-15,
                        threshold_limit=-15,
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

        weight = np.max([0, self.beta])

        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (r1, r2, r3), (lp1, lp2, lp3), logpy = output

        # reconstruction accuracies
        ll1 = torch.stack(
            [calc_ll(r, tgt1) for r in r1], dim=1
        )
        ll2 = torch.stack(
            [calc_ll(r, tgt2) for r in r2], dim=1
        )
        ll3 = torch.stack(
            [calc_ll(r, tgt3) for r in r3], dim=1
        )

        lp1 = lp1.log_softmax(dim=-1)
        lp2 = lp2.log_softmax(dim=-1)
        lp3 = lp3.log_softmax(dim=-1)

        llik = []

        for k, vals in knowledge.items():
            for v0, v1 in vals:
                
                weight1 = (torch.zeros_like(lp1[:, v0]) + lp1[:, v0]).detach() - lp1[:, v0]
                weight2 = (torch.zeros_like(lp2[:, v1]) + lp2[:, v1]).detach() - lp2[:, v1]
                weight3 = (torch.zeros_like(lp3[:, k])  + lp3[:, k]).detach()  - lp3[:, k]
                
                llik += [
                    (   
                        (ll1[:, v0] + weight1)+ 
                        (ll2[:, v1] + weight2)+
                        (ll3[:, k] + weight3) 
                        # ll3[:, k] + ll1[:, v0] + ll2[:, v1]
                        # (ll3[:, k] + weight3 * ll3[:, k]).detach() + weight3 * ll3[:, k] +
                        # (ll1[:, v0] + weight1 * ll1[:, v0]).detach() + weight1 * ll1[:, v0] +
                        # (ll2[:, v1] + weight2 * ll2[:, v1]).detach() + weight2 * ll2[:, v1]
                        # - lp3[:, k] - lp1[:, v0] - lp2[:, v1]
                    ) / 3
                ]

        llik = torch.stack(llik, dim=1)
        recon_losses, labels = llik.min(dim=1)

        loss = (logpy.exp() * (llik + logpy)).sum(dim=-1).mean()
        loss += weight*recon_losses.mean()
        loss += weight*F.nll_loss(logpy, labels)

        # llik = ((lp1.exp() * (ll1 + lp1)).sum(dim=-1) +
        #         (lp2.exp() * (ll2 + lp2)).sum(dim=-1) +
        #         (lp3.exp() * (ll3 + lp3)).sum(dim=-1))

        # recon_losses, labels = llik.min(dim=1)
        # loss = (logpy.exp() * (llik + logpy)).sum(dim=-1).mean()
        # loss += recon_losses.mean()

        return loss

    def warmup_hook(self, model, train_loader):
        # print("Warming up")
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # for epoch in range(0, 5):
        #     for batch_idx, ((in_data1, in_target1),
        #                     (in_data2, in_target2),
        #                     (in_data3, in_target3)) in enumerate(train_loader):
        #         targ_data1 = in_data1.reshape(len(in_data1), -1).to(self.device)
        #
        #         r = model(targ_data1, warmup=True)
        #         loss = F.binary_cross_entropy_with_logits(r, targ_data1, reduction="none").sum(dim=1).mean()
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        # print("Warmup Complete")
        pass

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

        vals = (
            torch.tensor([k for k, vals in knowledge.items() for v in vals])[None, :]
            .repeat(len(logpy), 1)
            .to(self.device)
        )
        acc = (
            (vals[np.arange(len(logpy)), logpy.argmax(dim=1)] == lbl3).float()
        ).tolist()

        self.losses["loss"].update(loss.data.item(), tgt1.size(0))
        self.losses["accuracy"].update(acc, tgt3.size(0))
        self.losses["entropy"].update(
            (-(logpy.exp() * logpy).sum(dim=1)).mean().data.item(),
            tgt3.size(0),
        )

    def epoch_finished_hook(self, epoch, model, val_loader):
        if self.device == "cpu":
            self.plot_model_samples(epoch, model)
        self.beta -= .05

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
            f'{type} [{epoch+1}/{self.epochs}]: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc {round(self.losses["accuracy"].avg, 3)} \t'
            f'Ent {round(self.losses["entropy"].avg, 3)} \n'
        )

        print(text, end="")
        self.logfile.write(text + "\n")


mnist_experiment_options = {
    "mnist_base": BaseMNISTExperiment,
    "mnist_with_constraints": ConstrainedMNIST,
}

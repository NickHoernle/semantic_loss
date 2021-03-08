import torch.nn.functional as F

from symbolic import train
from symbolic.utils import *
from experiment.datasets import *
from experiment.generative import MnistVAE
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
        (recons1, recons2, recons3), (lp1, lp2, lp3) = output
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
        (recons1, recons2, recons3), (lp1, lp2, lp3) = output

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
            -(
                (lp1.exp() * lp1).sum(dim=1)
                - (lp2.exp() * lp2).sum(dim=1)
                - (lp3.exp() * lp3).sum(dim=1)
            ).mean().data.item(),
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

    def iter_done(self, type="Train"):
        text = (
            f'{type}: Loss {round(self.losses["loss"].avg, 3)}\t '
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


def calc_ll(params, target):
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

    return rcon + kld


class ConstrainedMNIST(BaseMNISTExperiment):
    def __init__(
        self,
        **kwargs,
    ):
        kwargs["sloss"] = True
        super().__init__(**kwargs)

    def criterion(self, output, target, train=True):

        (tgt1, tgt2, tgt3), (lbl1, lbl2, lbl3) = target
        (recons1, recons2, recons3), (lp1, lp2, lp3) = output
        ll = []
        logpy = []

        for i, vals in knowledge.items():
            for j, v in enumerate(vals):
                ll1 = calc_ll(recons1[v[0]], tgt1)
                ll2 = calc_ll(recons2[v[1]], tgt2)
                ll3 = calc_ll(recons3[i], tgt3)

                ll += [ll1 + ll2 + ll3]
                logpy += [lp1[:, v[0]] + lp2[:, v[1]] + lp3[:, i]]

        preds = torch.stack(ll, dim=1)
        logpy = torch.stack(logpy, dim=1).log_softmax(dim=1)

        return (logpy.exp() * (preds + logpy)).sum(dim=1).mean()


mnist_experiment_options = {
    "mnist_base": BaseMNISTExperiment,
    "mnist_with_constraints": ConstrainedMNIST,
}

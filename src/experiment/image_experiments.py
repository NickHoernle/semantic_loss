import torch.nn.functional as F

from symbolic import symbolic
from symbolic import train
from symbolic.utils import *
from experiment.datasets import *
from experiment.constrainedwideresnet import ConstrainedModel
from experiment.wideresnet import WideResNet
from experiment.class_mapping import *


class BaseImageExperiment(train.Experiment):
    """Experimental setup for training with domain knowledge specified by a DNF logic formula on the CIFAR10 and CIFAR100 datasets. Wraps: train.Experiment.

    Image Experiment Parameters:
        lower_limit     lower limit for the logic thresholds that are applied
        upper_limit     uppser limit for the logic thresholds that are applied
        layers          number layers in WideResNet
        widen_factor    widen factor in WideResNet
        augment         apply standard augmentation to the data (flips, rotations to images)
        superclass      to use the superclass accuracy
        name            name to use in logging the results
    \n
    """

    __doc__ += train.Experiment.__doc__

    def __init__(
        self,
        lower_limit: float = -10.0,
        upper_limit: float = -2.0,
        layers: int = 28,
        widen_factor: int = 10,
        augment: bool = True,
        sloss: bool = False,
        superclass: bool = False,
        name: str = "WideResNet",
        **kwargs,
    ):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.layers = layers
        self.widen_factor = widen_factor
        self.augment = augment
        self.sloss = sloss
        self.superclass = superclass
        self.name = name

        self.classes = []
        self.class_mapping_ = None
        self.class_idxs_ = []

        super().__init__(**kwargs)

    @property
    def class_mapping(self):
        if type(self.class_mapping_) == type(None):
            self.class_mapping_ = torch.eye(len(self.classes)).to(self.device)
            for idxs in self.class_idxs:
                for ix in idxs:
                    self.class_mapping_[ix, idxs[0]] = 1.0
        return self.class_mapping_

    @property
    def params(self):
        return f"{self.name}-{self.lr}_{self.seed}-{self.layers}-{self.widen_factor}"

    @property
    def class_idxs(self):
        if len(self.class_idxs_) == 0:
            self.class_idxs_ = [t.ixs1 for t in self.logic_terms]
        return self.class_idxs_

    def get_loaders(self):
        train_loader, val_loader, classes = get_train_valid_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            augment=self.augment,
            random_seed=self.seed,
            valid_size=0.1,
            shuffle=True,
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        test_loader = get_test_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            dataset=self.dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.classes = classes

        return train_loader, val_loader, test_loader

    def create_model(self):
        if self.sloss:
            return ConstrainedModel(
                self.layers,
                self.num_classes,
                self.logic_terms,
                self.widen_factor,
                dropRate=self.droprate,
            )

        if self.superclass:
            return WideResNet(
                self.layers,
                self.num_super_classes,
                self.widen_factor,
                dropRate=self.droprate,
            )
        return WideResNet(
            self.layers, self.num_classes, self.widen_factor, dropRate=self.droprate
        )

    def get_optimizer_and_scheduler(self, model, train_loader):
        optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(train_loader) * self.epochs
        )
        return optimizer, scheduler

    def init_meters(self):
        loss = AverageMeter()
        accuracy = AccuracyMeter()
        superclass_accuracy = AccuracyMeter()
        self.losses = {
            "loss": loss,
            "accuracy": accuracy,
            "superclass_accuracy": superclass_accuracy,
        }

    def get_input_data(self, data):
        input_imgs, targets = data
        input_imgs = input_imgs.to(self.device)
        return input_imgs

    def get_target_data(self, data):
        input_imgs, targets = data
        targets = targets.to(self.device)
        return targets

    def criterion(self, output, target, train=True):
        if self.sloss and train:
            class_preds, logic_preds = output
            ll = []
            for j, p in enumerate(class_preds.split(1, dim=1)):
                y_onehot = torch.zeros_like(p.squeeze(1))
                y_onehot.scatter_(1, target[:, None], 1)
                y_onehot = y_onehot.mm(self.class_mapping)
                ll += [
                    F.binary_cross_entropy_with_logits(
                        p.squeeze(1), y_onehot, reduction="none"
                    ).sum(dim=1)
                ]

            pred_loss = torch.stack(ll, dim=1)
            recon_losses, labels = pred_loss.min(dim=1)

            loss = (logic_preds.exp() * (pred_loss + logic_preds)).sum(dim=1).mean()
            loss += recon_losses.mean()
            loss += F.nll_loss(logic_preds, labels)

        elif self.superclass:
            class_pred = output
            new_tgts = torch.zeros_like(target)
            for j, ixs in enumerate(self.class_idxs[1:]):
                new_tgts += (j + 1) * (
                    torch.stack([target == k for k in ixs], dim=1).any(dim=1)
                ).long()
            target = new_tgts
            loss = F.cross_entropy(class_pred, target)

        else:
            class_preds = output
            loss = F.cross_entropy(class_preds, target)

        return loss

    def update_train_meters(self, loss, output, target):
        self.losses["loss"].update(loss.data.item(), target.size(0))

        if self.sloss:
            cp, logic_preds = output
            ixs = np.arange(target.size(0))
            class_preds = cp[ixs, logic_preds.argmax(dim=1)]
        else:
            class_preds = output

        self.losses["accuracy"].update(
            (class_preds.argmax(dim=1) == target).tolist(), target.size(0)
        )

        if not self.superclass:
            new_tgts = torch.zeros_like(target)
            for i, ixs in enumerate(self.class_idxs[1:]):
                new_tgts += (i + 1) * (
                    torch.stack([target == i for i in ixs], dim=1).any(dim=1)
                )

            if self.sloss:
                self.losses["superclass_accuracy"].update(
                    (logic_preds.argmax(dim=1) == new_tgts).tolist(),
                    target.data.shape[0],
                )
            else:
                forward_mapping = [int(c) for ixs in self.class_idxs for c in ixs]

                split = class_preds.softmax(dim=1)[:, forward_mapping].split(
                    [len(i) for i in self.class_idxs], dim=1
                )
                new_pred = torch.stack([s.sum(dim=1) for s in split], dim=1)
                self.losses["superclass_accuracy"].update(
                    (new_pred.data.argmax(dim=1) == new_tgts).tolist(),
                    output.data.shape[0],
                )

    def update_test_meters(self, loss, output, target):
        self.losses["loss"].update(loss.data.item(), target.size(0))
        self.losses["accuracy"].update(
            (output.data.argmax(dim=1) == target).tolist(), target.size(0)
        )

        if not self.superclass:
            new_tgts = torch.zeros_like(target)
            for i, ixs in enumerate(self.class_idxs[1:]):
                new_tgts += (i + 1) * (
                    torch.stack([target == i for i in ixs], dim=1).any(dim=1)
                )
            forward_mapping = [int(c) for ixs in self.class_idxs for c in ixs]

            split = output.softmax(dim=1)[:, forward_mapping].split(
                [len(i) for i in self.class_idxs], dim=1
            )
            new_pred = torch.stack([s.sum(dim=1) for s in split], dim=1)

            self.losses["superclass_accuracy"].update(
                (new_pred.data.argmax(dim=1) == new_tgts).tolist(), output.data.shape[0]
            )

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f'Loss {self.losses["loss"].val:.4f} ({self.losses["loss"].avg:.4f})\t'
            f'Acc {self.losses["accuracy"].val:.4f} ({self.losses["accuracy"].avg:.4f})\t'
            f'AccSC {self.losses["superclass_accuracy"].val:.4f} ({self.losses["superclass_accuracy"].avg:.4f}\n)'
        )

    def iter_done(self, type="Train"):
        text = (
            f'{type}: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc {round(self.losses["accuracy"].avg, 3)}\t'
            f'AccSC {round(self.losses["superclass_accuracy"].avg, 3)}\n'
        )
        self.logfile.write(text)
        print(text, end="")

    def update_best(self, val):
        if val < self.best_loss:
            self.best_loss = val
            return True
        return False


class Cifar10Experiment(BaseImageExperiment):
    def __init__(self, **kwargs):
        self.dataset = "cifar10"
        self.num_classes = 10
        self.num_super_classes = 2

        super().__init__(**kwargs)

    @property
    def logic_terms(self):
        return [
            symbolic.GEQConstant(
                ixs1=[0, 1, 8, 9],
                ixs_less_than=[2, 3, 4, 5, 6, 7],
                ixs_not=[],
                threshold_upper=self.upper_limit,
                threshold_lower=self.upper_limit - 5,
                threshold_limit=self.upper_limit,
                device=self.device,
            ),
            symbolic.GEQConstant(
                ixs1=[2, 3, 4, 5, 6, 7],
                ixs_less_than=[0, 1, 8, 9],
                ixs_not=[],
                threshold_upper=self.upper_limit,
                threshold_lower=self.upper_limit - 5,
                threshold_limit=self.upper_limit,
                device=self.device,
            ),
        ]


class Cifar100Experiment(BaseImageExperiment):
    def __init__(self, **kwargs):
        self.dataset = "cifar100"
        self.num_classes = 100
        self.num_super_classes = 20

        super().__init__(**kwargs)

    @property
    def logic_terms(self):
        idxs = [
            [i for i, c in enumerate(self.classes) if superclass_mapping[c] == label]
            for label, ix in sorted(super_class_label.items(), key=lambda x: x[1])
        ]

        terms = []
        for i, ixs in enumerate(idxs):
            all_idsx = np.arange(len(self.classes))
            not_idxs = all_idsx[~np.isin(all_idsx, ixs)].tolist()
            terms += [
                symbolic.GEQConstant(
                    ixs1=ixs,
                    ixs_less_than=not_idxs,
                    ixs_not=[],
                    threshold_upper=self.upper_limit,
                    threshold_lower=self.upper_limit - 5,
                    threshold_limit=self.lower_limit,
                    device=self.device,
                )
            ]

        return terms


image_experiment_options = {
    "cifar10": Cifar10Experiment,
    "cifar100": Cifar100Experiment,
}

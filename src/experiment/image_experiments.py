import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../../../dl2'))

import torch.nn.functional as F

from symbolic import symbolic
from symbolic import train
from symbolic.utils import *
from experiment.datasets import *
from experiment.constrainedwideresnet import ConstrainedModel
from experiment.wideresnet import WideResNet, HierarchicalModel
from experiment.class_mapping import *
import torch.nn as nn

import dl2lib as dl2

# This is to shoe horn the default DL2 args into this project for the baseline tests.
import argparse
parser = argparse.ArgumentParser(description='desc')
parser.add("--use-eps", default=False, required=False, help="use the +epsilon translation for strict inequalities")
parser.add_argument("--eps-check", type=float, default=0, required=False)
parser.add_argument('--or', default='min', type=str)
dl2_args = parser.parse_known_args()[0]


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
        lower_limit: float = -5.89,
        upper_limit: float = 0,
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

        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

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
        train_loader, val_loader, classes, _ = get_train_valid_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            augment=self.augment,
            random_seed=self.seed,
            valid_size=0.1,
            shuffle=True,
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            do_normalize=True,
        )

        test_loader = get_test_loader(
            data_dir=self.dataset_path,
            batch_size=self.batch_size,
            dataset=self.dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            do_normalize=True
        )

        self.classes = classes

        return train_loader, val_loader, test_loader

    def create_model(self):
        pass

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
        constraint = AccuracyMeter()
        self.losses = {
            "loss": loss,
            "accuracy": accuracy,
            "superclass_accuracy": superclass_accuracy,
            "constraint": constraint
        }

    def get_val_loss(self):
        return self.losses["accuracy"].avg

    def update_best(self, val):
        if val > self.best_loss:
            self.best_loss = val
            return True
        return False

    def get_input_data(self, data):
        input_imgs, targets = data
        input_imgs = input_imgs.to(self.device)
        return input_imgs

    def get_target_data(self, data):
        input_imgs, targets = data
        targets = targets.to(self.device)
        new_tgts = torch.zeros_like(targets)

        for j, ixs in enumerate(self.class_idxs[1:]):
            new_tgts += (j + 1) * (
                torch.stack([targets == k for k in ixs], dim=1).any(dim=1)
            ).long().to(self.device)

        return (targets, new_tgts)

    def criterion(self, output, targets, train=True):
        pass

    def update_train_meters(self, loss, output, targets):
        (target, sc_target) = targets
        self.losses["loss"].update(loss.data.item(), target.size(0))

    def update_test_meters(self, loss, output, targets):
        self.update_train_meters(loss, output, targets)

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f'Loss {self.losses["loss"].val:.4f} ({self.losses["loss"].avg:.4f})\t'
            f'Acc {self.losses["accuracy"].val:.4f} ({self.losses["accuracy"].avg:.4f})\t'
            f'AccSC {self.losses["superclass_accuracy"].val:.4f} ({self.losses["superclass_accuracy"].avg:.4f}\n)'
        )

    def iter_done(self, epoch, type="Train"):
        text = (
            f'{type} [{epoch+1}/{self.epochs}]: Loss {round(self.losses["loss"].avg, 3)}\t '
            f'Acc {round(self.losses["accuracy"].avg, 3)}\t'
            f'AccSC {round(self.losses["superclass_accuracy"].avg, 3)}\t'
            f'ConstraintAcc {round(self.losses["constraint"].avg, 3)}\n'
        )
        self.logfile.write(text)
        print(text, end="")

    def run_validation(self, epoch):
        if epoch % 5 == 0:
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


class Cifar100Base(BaseImageExperiment):
    def __init__(self, **kwargs):
        self.dataset = "cifar100"
        self.num_classes = 100
        self.num_super_classes = 20
        super().__init__(**kwargs)
        self.idxs_ = []

    @property
    def class_index_mapping(self):
        if len(self.idxs_) == 0:
            self.idxs_ = [
                [i for i, c in enumerate(self.classes) if superclass_mapping[c] == label]
                for label, ix in sorted(super_class_label.items(), key=lambda x: x[1])
            ]
        return self.idxs_

    @property
    def logic_terms(self):

        terms = []
        for i, ixs in enumerate(self.class_index_mapping):
            all_idsx = np.arange(len(self.classes))
            not_idxs = all_idsx[~np.isin(all_idsx, ixs)].tolist()
            terms += [
                symbolic.GEQConstant(
                    ixs1=ixs,
                    ixs_less_than=not_idxs,
                    ixs_not=[],
                    # threshold_upper=0,
                    more_likely_multiplier=10,
                    # threshold_lower=-5.89,
                    device=self.device,
                )
            ]

        return terms


class Cifar100Experiment(Cifar100Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Cifar100-MultiplexNet"

    def create_model(self):
        return ConstrainedModel(
            self.layers,
            self.num_classes,
            self.logic_terms,
            self.widen_factor,
            dropRate=self.droprate,
        )

    def criterion(self, output, targets, train=True):
        (target, sc_target) = targets

        class_preds, logic_preds = output
        ll = []
        for j, logic_branch in enumerate(class_preds.split(1, dim=1)):
            ll += [F.cross_entropy(logic_branch.squeeze(1), target, reduction="none")]

        pred_loss = torch.stack(ll, dim=1)

        loss = (logic_preds.exp() * (pred_loss + logic_preds)).sum(dim=1).mean()

        return loss

    def update_train_meters(self, loss, output, targets):
        (target, sc_target) = targets

        cp, logic_preds = output
        ixs = np.arange(target.size(0))
        class_preds = cp[ixs, logic_preds.argmax(dim=1)]

        self.losses["accuracy"].update(
            (class_preds.argmax(dim=1) == target).tolist(), target.size(0)
        )

        self.losses["superclass_accuracy"].update(
            (logic_preds.argmax(dim=1) == sc_target).tolist(),
            target.data.shape[0],
        )

        valid_constraints = [t.valid(class_preds) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        self.losses["constraint"].update(v_c.tolist(), v_c.size(0))

        super(Cifar100Experiment, self).update_train_meters(loss, output, targets)

    def epoch_finished_hook(self, epoch, model, val_loader):
        pass


class VanillaBaseline(Cifar100Base):
    def __init__(self, **kwargs):
        self.name = "Cifar100-VanillaBaseline"
        super().__init__(**kwargs)

    def create_model(self):
        return WideResNet(
            self.layers,
            self.num_classes,
            self.widen_factor,
            dropRate=self.droprate,
        )

    def criterion(self, output, targets, train=True):
        (target, sc_target) = targets
        return self.loss_criterion(output, target)

    def update_train_meters(self, loss, output, targets):
        (target, sc_target) = targets

        self.losses["accuracy"].update(
            (output.argmax(dim=1) == target).tolist(), target.size(0)
        )

        forward_mapping = [int(c) for ixs in self.class_idxs for c in ixs]

        split = output.softmax(dim=1)[:, forward_mapping].split(
            [len(i) for i in self.class_idxs], dim=1
        )
        new_pred = torch.stack([s.sum(dim=1) for s in split], dim=1)
        self.losses["superclass_accuracy"].update(
            (new_pred.data.argmax(dim=1) == sc_target).tolist(),
            output.data.shape[0],
        )

        valid_constraints = [t.valid(output) for t in self.logic_terms]
        v_c = torch.stack(valid_constraints, dim=1).any(dim=1)
        self.losses["constraint"].update(v_c.tolist(), v_c.size(0))

        super(VanillaBaseline, self).update_train_meters(loss, output, targets)


class SuperclassOnly(Cifar100Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Cifar100-SuperclassOnly"

    def create_model(self):
        return WideResNet(
            self.layers,
            self.num_super_classes,
            self.widen_factor,
            dropRate=self.droprate,
        )

    def criterion(self, output, targets, train=True):
        (target, sc_target) = targets
        return self.loss_criterion(output, sc_target)

    def update_train_meters(self, loss, output, targets):
        (target, sc_target) = targets
        self.losses["accuracy"].update(
            (output.argmax(dim=1) == sc_target).tolist(), target.size(0)
        )
        super().update_train_meters(loss, output, targets)


class HierarchicalBaseline(Cifar100Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Cifar100-HierarchicalBaseline"

    def create_model(self):
        return HierarchicalModel(
            self.layers, self.num_classes, self.widen_factor, dropRate=self.droprate
        )

    def get_target_data(self, data):
        input_imgs, targets = data
        targets = targets.to(self.device)

        class_pred_targets = torch.ones_like(targets)
        superclass_pred_targets = torch.ones_like(targets)

        tgts = targets.tolist()

        for i, t in enumerate(tgts):
            sc_label, c_label = hierarchical_label_structure[self.classes[t]]
            class_pred_targets[i] *= c_label
            superclass_pred_targets[i] *= sc_label

        return class_pred_targets, superclass_pred_targets

    def criterion(self, output, targets, train=True):
        (target, sc_target) = targets
        class_pred, sc_pred = output

        loss = F.nll_loss(sc_pred, sc_target)
        loss += F.nll_loss(class_pred, target)

        return loss

    def update_train_meters(self, loss, output, targets):
        (target, sc_target) = targets
        class_pred, sc_pred = output

        self.losses["accuracy"].update(
            (
                (class_pred.argmax(dim=1) == target) & (sc_pred.argmax(dim=1) == sc_target)
             ).tolist(), target.size(0)
        )

        self.losses["superclass_accuracy"].update(
            (sc_pred.argmax(dim=1) == sc_target).tolist(),
            target.size(0),
        )

        super().update_train_meters(loss, output, targets)


class DL2Baseline(VanillaBaseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Cifar100-DL2Baseline"
        self.ratio = 1.0
        self.constraint_weight = 0.6
        self.increase_constraint_weight = 1.0

    def criterion(self, output, targets, train=True):
        (target, sc_target) = targets

        dl2_one_group = []

        eps = .01
        probs_u = output.softmax(dim=1)

        for i, ixs in enumerate(self.class_index_mapping):
            gsum = probs_u[:, ixs].sum(dim=1)
            dl2_one_group.append(dl2.Or([dl2.GT(gsum, 1.0 - eps), dl2.LT(gsum, eps)]))

        dl2_one_group = dl2.And(dl2_one_group)
        dl2_loss = dl2_one_group.loss(dl2_args).mean()

        return self.loss_criterion(output, target) + (self.constraint_weight * self.increase_constraint_weight) * dl2_loss

    def epoch_finished_hook(self, epoch, model, val_loader):
        self.increase_constraint_weight = self.ratio ** epoch # see https://github.com/eth-sri/dl2/blob/9842cdf2b145c24481eb81e13ed66b2600f196fc/training/semisupservised/main.py#L316


image_experiment_options = {
    "cifar100_multiplexnet": Cifar100Experiment,
    "cifar100_baseline_full": VanillaBaseline,
    "cifar100_baseline_superclass_only": SuperclassOnly,
    "cifar100_hierarchical_baseline": HierarchicalBaseline,
    "cifar100_dl2_baseline": DL2Baseline,
}

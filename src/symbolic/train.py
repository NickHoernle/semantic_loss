import time
import git
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from symbolic.utils import *
from symbolic.symbolic import *

# used for logging to TensorBoard
device = "cuda" if torch.cuda.is_available() else "cpu"


class Experiment(ABC):
    @initializer
    def __init__(
        self,
        dataset: str = "cifar10",
        dataset_path: str = "../data",
        checkpoint_dir: str = "runs",
        epochs: int = 200,
        seed: int = 12,
        start_epoch: int = 0,
        batch_size: int = 256,
        lower_limit: float = -10.0,
        upper_limit: float = -2.0,
        learning_rate: float = 1e-1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        print_freq: int = 10,
        layers: int = 28,
        widen_factor: int = 10,
        droprate: float = 0.0,
        augment: bool = True,
        resume: bool = False,
        name: str = "WideResNet",
        tensorboard: bool = False,
        sloss: bool = True,
        superclass: bool = False,
    ):
        """
        Creates the experiment object that contains all of the experiment configuration parameters
        """
        super().__init__()
        self.git_commit = ""
        self.device = None
        self.start_epoch = 0
        self.losses = AverageMeter()
        pass

    @property
    def lr(self):
        return self.learning_rate

    @property
    def num_workers(self):
        if device == "cpu":
            return 4
        return 1

    @property
    def pin_memory(self):
        if device == "cpu":
            return False
        return True

    @property
    def params(self):
        return f"{self.name}-{self.layers}-{self.widen_factor}_{self.lr}_{self.seed}"

    @property
    def checkpoint_directory(self):
        assert self.git_commit != ""
        return os.path.join(self.checkpoint_dir, self.git_commit, self.params)

    @property
    def best_checkpoint(self):
        return os.path.join(self.checkpoint_directory, "best_checkpoint.pt")

    @property
    def checkpoint(self):
        return os.path.join(self.checkpoint_directory, "checkpoint.pt")

    def init_meters(self):
        self.losses = AverageMeter()

    def update_train_meters(self, loss, model_output, targets):
        self.losses.update(loss.data.item(), model_output.size(0))

    def update_test_meters(self, loss, model_output, targets):
        self.losses.update(loss.data.item(), model_output.size(0))

    @abstractmethod
    def get_loaders(self):
        """
        Returns the train, valid and test loaders used for executing the experiment
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Creates the model that will be used for training, testing and running in this experiment
        """
        pass

    @abstractmethod
    def criterion(self, output, target):
        """
        Returns the loss function that will be used in the training and evaluation loops
        """
        pass

    @abstractmethod
    def get_optimizer_and_scheduler(self, model):
        """
        Returns the optimizer that will be used in the training loops
        """
        pass

    @abstractmethod
    def get_input_data(self, data):
        pass

    @abstractmethod
    def get_target_data(self, data):
        pass

    def log(self, epoch, batch_time):
        print(
            f"Epoch: [{0}][{1}/{2}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f"Loss {self.losses.val:.4f} ({self.losses.avg:.4f})"
        )

    def iter_done(self, type="Train"):
        print(f"{type}: Loss {round(self.losses.avg, 3)}")


def main(experiment):

    print(experiment.params)

    repo = git.Repo(search_parent_directories=True)
    # set the git commit for logging purposes
    experiment.git_commit = repo.head.object.hexsha
    experiment.device = device

    # if args.tensorboard: configure(os.path.join(args.checkpoint_dir, git_commit, params))
    # Data loading code
    train_loader, val_loader, test_loader = experiment.get_loaders()

    # create model
    model = experiment.create_model()

    # get the number of model parameters
    print(f"Number of model parameters: {sum([p.numel() for p in model.parameters()])}")

    # no current support for parallel GPU execution
    model = model.to(device)

    # optionally resume from a checkpoint
    if experiment.resume:
        if os.path.isfile(experiment.checkpoint):
            print(f"=> loading checkpoint from '{experiment.checkpoint}'")
            checkpoint = torch.load(experiment.checkpoint)
            experiment.start_epoch = checkpoint["epoch"]
            experiment.best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            print(
                f"=> loaded checkpoint '{experiment.checkpoint}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{experiment.checkpoint}'")

    # TODO: why do I need this again?
    cudnn.benchmark = True

    optimizer, scheduler = experiment.get_optimizer_and_scheduler(model, train_loader)

    for epoch in range(experiment.start_epoch, experiment.epochs):

        # train for one epoch
        # train(train_loader, model, optimizer, scheduler, epoch, experiment)

        # evaluate on validation set
        val1 = validate(val_loader, model, epoch, experiment)

        # remember best prec@1 and save checkpoint
        is_best = experiment.update_best(val1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": experiment.best_loss,
            },
            is_best,
            experiment,
        )
    print("Best loss: ", experiment.best_loss)

    # load the best model and evaluate on the test set
    print("======== TESTING ON UNSEEN DATA =========")
    print("======== USE FINAL MODEL =========")
    prec1 = validate(test_loader, model, 0, experiment)
    print("Final Model accuracy ====> ", prec1)
    print("======== USE BEST MODEL =========")
    checkpoint = torch.load(experiment.best_checkpoint_directory)
    model.load_state_dict(checkpoint["state_dict"])
    prec1 = validate(test_loader, model, 0, experiment)
    print("Test accuracy ====> ", prec1)


def train(train_loader, model, optimizer, scheduler, epoch, experiment):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    experiment.init_meters()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        model_input = experiment.get_input_data(data)
        target = experiment.get_target_data(data)

        model_output = model(model_input)
        loss = experiment.criterion(model_output, target)

        experiment.update_train_meters(loss, model_output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % experiment.print_freq == experiment.print_freq - 1:
            experiment.log(epoch, batch_time)

    experiment.iter_done(type="Train")

    # TODO: setup tensorboard
    # log to TensorBoard
    # if experiment.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, epoch, experiment):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    experiment.init_meters()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        model_input = experiment.get_input_data(data)
        target = experiment.get_target_data(data)

        # compute output
        with torch.no_grad():
            output = model(model_input, test=True)

        loss = experiment.criterion(output, target, train=False)
        experiment.update_test_meters(loss, output, target)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % experiment.print_freq == experiment.print_freq - 1:
            experiment.log(epoch, batch_time)

    experiment.iter_done(type="Test")
    # TODO: setup tensorboard
    # log to TensorBoard
    # if args.tensorboard:
    #     from tensorboard_logger import configure, log_value
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    # return top1.avg

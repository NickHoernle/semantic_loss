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
    """Experimental setup for training with domain knowledge specified by a DNF logic formula.

    Base Experiment Parameters:
        dataset_path    path to the data
        checkpoint_dir  directory where the checkpoints and logs will be stored
        epochs          number of epochs for training
        seed            random seed for initialising experiments
        start_epoch     start the training from this epoch (default = 0)
        batch_size      minibatch size for training
        learning_rate   learning rate for training
        momentum        momentum for SGD
        weight_decay    weight decay param in SGD
        print_freq      how often to log results
        droprate        dropout rate to use
        resume          resume training from checkpoint
        tensorboard     to use tensorboard
    """

    def __init__(
        self,
        dataset_path: str = "../data",
        checkpoint_dir: str = "runs",
        epochs: int = 200,
        seed: int = 12,
        start_epoch: int = 0,
        batch_size: int = 256,
        learning_rate: float = 1e-1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        print_freq: int = 10,
        droprate: float = 0.0,
        resume: bool = False,
        tensorboard: bool = False,
    ):
        """
        Creates the experiment object that contains all of the experiment configuration parameters
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.seed = seed
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.droprate = droprate
        self.resume = resume
        self.tensorboard = tensorboard

        self.git_commit = ""
        self.device = None
        self.start_epoch = 0
        self.losses = AverageMeter()
        self.best_loss = np.infty

        main(self)

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
        return f"{self.name}_{self.lr}_{self.seed}"

    @property
    def checkpoint_directory(self):
        assert self.git_commit != ""
        path = os.path.join(self.checkpoint_dir, self.git_commit, self.params)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    @property
    def figures_directory(self):
        figs_dir = os.path.join(self.checkpoint_directory, "figures")
        if not os.path.exists(figs_dir):
            os.mkdir(figs_dir)
        return figs_dir

    @property
    def logs_directory(self):
        logs_dir = os.path.join(self.checkpoint_directory, "logs")
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        return logs_dir

    @property
    def best_checkpoint(self):
        return os.path.join(self.checkpoint_directory, "best_checkpoint.pt")

    @property
    def checkpoint(self):
        return os.path.join(self.checkpoint_directory, "checkpoint.pt")

    @property
    def logfile(self):
        if type(self.logfile_) == type(None):
            self.logfile_ = open(
                os.path.join(self.logs_directory, "logs.txt"), "w", buffering=1
            )
        return self.logfile_

    def init_meters(self):
        self.losses = AverageMeter()

    def update_train_meters(self, loss, model_output, targets):
        self.losses.update(loss.data.item(), model_output.size(0))

    def update_test_meters(self, loss, model_output, targets):
        self.losses.update(loss.data.item(), model_output.size(0))

    def pre_train_hook(self, *args, **kwargs):
        pass

    def post_train_hook(self, *args, **kwargs):
        pass

    def epoch_finished_hook(self, *args, **kwargs):
        pass

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

    def log(self, text):
        self.logfile.write(f"{text}\n")

    def log_iter(self, epoch, batch_time):
        self.logfile.write(
            f"Epoch: [{epoch}/{self.epochs}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        )

    def iter_done(self, type="Train"):
        self.logfile.write(f"{type}: Loss {round(self.losses.avg, 3)}")


def main(experiment):

    repo = git.Repo(search_parent_directories=True)
    # set the git commit for logging purposes
    experiment.git_commit = repo.head.object.hexsha
    experiment.device = device

    # if args.tensorboard: configure(os.path.join(args.checkpoint_dir, git_commit, params))
    # Data loading code
    train_loader, val_loader, test_loader = experiment.get_loaders()

    # create model
    model = experiment.create_model()

    experiment.log(f"Starting experiment with params: {experiment.params}")

    # get the number of model parameters
    experiment.log(
        f"Number of model parameters: {sum([p.numel() for p in model.parameters()])}"
    )

    # no current support for parallel GPU execution
    model = model.to(device)

    # optionally resume from a checkpoint
    if experiment.resume:
        if os.path.isfile(experiment.checkpoint):
            experiment.log(f"=> loading checkpoint from '{experiment.checkpoint}'")
            checkpoint = torch.load(experiment.checkpoint)
            experiment.start_epoch = checkpoint["epoch"]
            experiment.best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            experiment.log(
                f"=> loaded checkpoint '{experiment.checkpoint}' (epoch {checkpoint['epoch']})"
            )
        else:
            experiment.log(f"=> no checkpoint found at '{experiment.checkpoint}'")

    # TODO: why do I need this again?
    cudnn.benchmark = True

    optimizer, scheduler = experiment.get_optimizer_and_scheduler(model, train_loader)

    experiment.pre_train_hook(train_loader)

    for epoch in range(experiment.start_epoch, experiment.epochs):

        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, experiment)

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
        experiment.epoch_finished_hook(epoch, model, val_loader)

    experiment.post_train_hook()

    all_results_file = open(
        os.path.join(experiment.checkpoint_dir, "results.txt"), "a", buffering=1
    )

    experiment.log(f"Best loss: {experiment.best_loss}")

    final_model_val_acc = validate(test_loader, model, 0, experiment)
    checkpoint = torch.load(experiment.best_checkpoint_directory)
    model.load_state_dict(checkpoint["state_dict"])
    best_model_val_acc = validate(test_loader, model, 0, experiment)

    experiment.log("======== TESTING ON UNSEEN DATA =========")
    experiment.log("======== USE FINAL MODEL =========")
    experiment.log(f"Final Model accuracy ====> {final_model_val_acc}")
    experiment.log("======== USE BEST MODEL =========")
    experiment.log(f"Final Model accuracy ====> {best_model_val_acc}")

    all_results_file.write(f"{experiment.params}: {best_model_val_acc}")
    all_results_file.close()
    experiment.logfile.close()


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
            experiment.log_iter(
                f"Epoch: [{0}][{1}/{2}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {experiment.losses.val:.4f} ({experiment.losses.avg:.4f})"
            )

    experiment.iter_done(type="Train")

    # TODO: setup tensorboard
    # log to TensorBoard
    # if experiment.tensorboard:
    #     log_value('train_loss', losses.avg, epoch)
    #     log_value('train_acc', top1.avg, epoch)


def validate(val_loader, model, epoch, experiment, logfile):
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

    experiment.iter_done(type="Test ")

    return loss
    # TODO: setup tensorboard
    # log to TensorBoard
    # if args.tensorboard:
    #     from tensorboard_logger import configure, log_value
    #     log_value('val_loss', losses.avg, epoch)
    #     log_value('val_acc', top1.avg, epoch)
    # return top1.avg

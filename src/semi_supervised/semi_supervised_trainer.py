"""
Author: Nick Hoernle
Define semi-supervised class for training generative models with both a labeled and an 
unlabeled loss
"""
import numpy as np

from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils.generative_trainer import GenerativeTrainer
from utils.logging import AverageMeter
from utils.data import get_samplers

from .datasets import get_semi_supervised

params = {
    "MNIST":{"num_categories": 10},
    "CIFAR10":{"num_categories": 10},
    "CIFAR100":{"num_categories": 100},
}


class SemiSupervisedTrainer(GenerativeTrainer):
    """
    Implements a semi-supervised learning training class for MNIST, CIFAR10 and CIFAR100 data.
    """

    def __init__(
        self,
        model_builder,
        model_parameters,
        input_data,
        output_data,
        dataset="MNIST",
        max_grad_norm=1,
        num_epochs=100,
        batch_size=256,
        lr=1e-3,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        early_stopping_lim=50,
        additional_model_config_args=[],
        num_loader_workers=8,
        num_labelled=4000,
        name="base-model",
        disable_tqdm_print=True,
    ):
        self.num_labelled = num_labelled
        self.dataset = dataset
        model_parameters = {**model_parameters, **params[dataset]}
        super().__init__(
            model_builder,
            model_parameters,
            input_data=input_data,
            output_data=output_data,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            use_cuda=use_cuda,
            num_test_samples=num_test_samples,
            seed=seed,
            gamma=gamma,
            resume=resume,
            early_stopping_lim=early_stopping_lim,
            additional_model_config_args=additional_model_config_args,
            num_loader_workers=num_loader_workers,
            data_shuffle=False,
            name=name,
            disable_tqdm_print=disable_tqdm_print,
        )
        self.num_categories = model_parameters['num_categories']

    @torch.enable_grad()
    def train(self, epoch, net, optimizer, loader, **kwargs):
        """
        Train step of model returned by model_builder
        """
        device = self.device

        net.train()
        loss_meter = AverageMeter()
        sloss_meter = AverageMeter()

        opt_all, opt_encode, opt_decode = optimizer

        with tqdm(disable=self.tqdm_print) as progress_bar:
            for data_sample in loader:

                data_l, label_l = data_sample[0]
                data_u, label_u = data_sample[1]

                b_size = data_u.size(0)
                label = torch.full((b_size,), 1, device=device)

                # prepare the data
                data_u = data_u.to(device)
                data_l = data_l.to(device)
                label_l = label_l.to(device)

                labelled_results = net(data_l)
                unlabelled_results = net(data_u)

                labelled_loss = self.labeled_loss(data_l, label_l, **labelled_results)
                unlabelled_loss = self.unlabeled_loss(data_u, **unlabelled_results)

                loss = labelled_loss + unlabelled_loss

                # sloss_meter.update(loss_logic.item(), data_u.size(0))
                loss_meter.update(loss.item(), data_u.size(0))

                progress_bar.set_postfix(nll=loss_meter.avg, sloss=sloss_meter.avg)
                progress_bar.update(data_u.size(0))

                self.global_step += data_u.size(0)

                break
            
        print(f"===============> Epoch {epoch}; SLoss: {sloss_meter.avg}; NLL: {loss_meter.avg}")
        return loss_meter.avg

    @torch.no_grad()
    def test(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Test step of model returned by model_builder
        """

        return_accuracy = kwargs.get("return_accuracy", False)

        device = self.device

        net.eval()
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        with tqdm(disable=self.tqdm_print) as progress_bar:
            for data, labels in loaders:

                data = data.to(device)
                labels = labels.to(device)

                results = net(data)
                loss = self.labeled_loss(data, labels, **results)

                predictions = torch.argmax(results["x"], dim=1)
                correct_meter.update(torch.sum(predictions==labels), labels.size(0))

                loss_meter.update(loss.item(), data.size(0))

        print(f"===============> Epoch {epoch}; Accuracy: {correct_meter.avg}; NLL: {loss_meter.avg}")

        if return_accuracy:
            return loss_meter.avg, correct_meter.avg
        return loss_meter.avg

    @staticmethod
    def labeled_loss(*args, **kwargs):
        """
        Loss for the labeled data
        """
        raise NotImplementedError("Not implemented labeled loss")

    @staticmethod
    def unlabeled_loss(*args, **kwargs):
        """
        Loss for the unlabeled data
        """
        raise NotImplementedError("Not implemented unlabeled loss")

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        train_dataset, test_dataset, image_shape, num_classes = get_semi_supervised(
            self.input_data,
            dataset=self.dataset,
            num_labelled=self.num_labelled,
            augment=True
        )

        return train_dataset, test_dataset, num_classes

    def convert_to_one_hot(self, num_categories, labels):
        labels = torch.unsqueeze(labels, 1)
        one_hot = torch.FloatTensor(len(labels), num_categories).zero_().to(self.device)
        one_hot.scatter_(1, labels, 1)
        return one_hot

    def sample_examples(self, epoch, net):
        labels = torch.zeros(64, self.num_categories).to(self.device)
        labels[torch.arange(64), torch.arange(8).repeat(8)] = 1
        img_sample = net.sample_labelled(labels)
        save_image(img_sample, f'{self.figure_path}/sample_' + str(epoch) + '.png')

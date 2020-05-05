"""
Author: Nick Hoernle
Define semi-supervised class for training generative models with both a labeled and an 
unlabeled loss
"""
import os

import numpy as np
import copy

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

from utils.generative_trainer import GenerativeTrainer
from utils.logging import AverageMeter

from robotic_constraints.dataloader import NavigateFromTo
from rss_code_and_data import (DMP)
from robotic_constraints.utils import plot_trajectories


class RCTrainer(GenerativeTrainer):
    """
    Implements a semi-supervised learning training class for MNIST, CIFAR10 and CIFAR100 data.
    """

    def __init__(
        self,
        model_builder,
        model_parameters,
        input_data,
        output_data,
        max_grad_norm=1,
        num_epochs=100,
        batch_size=256,
        lr=1e-3,
        use_cuda=True,
        num_test_samples=256,
        seed=0,
        gamma=0.9,
        resume=False,
        backward=False,
        early_stopping_lim=50,
        additional_model_config_args=['backward'],
        num_loader_workers=8,
        name="base-model",
    ):
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
            data_shuffle=True,
            name=name,
        )
        self.data_dims = model_parameters['data_dim']
        self.backward = backward

    def train(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Train step of model returned by model_builder
        """
        device = self.device
        dims = self.data_dims
        net.train()
        loss_meter = AverageMeter()
        x = loaders.dataset.__getitem__(0)[0]
        dmp = DMP(x.size()[-1] // 2, dt=1 / 100, d=2, device=device)

        # import pdb
        # pdb.set_trace()

        with tqdm(total=loaders.dataset.__len__()) as progress_bar:
            for x, condition_params, trajectory in loaders:
                # del x_l
                # del condition_params

                x = x.to(device)
                trajectory = trajectory.to(device)
                # condition_params = condition_params.to(device)

                optimizer.zero_grad()
                results = net(x)
                recon_traj = self.get_y_tracks(results[0])
                loss_f = self.forward_loss((x, trajectory), *results, recon_traj)

                loss_b = 0
                if self.backward and epoch > 1:
                    # optimizer.zero_grad()
                    # num_samples = len(x)
                    # net_output = net.sample(num_samples)
                    # y_track = self.get_y_tracks(net_output)
                    loss_b = self.backward_loss(recon_traj, results, loaders.dataset.constraints)

                loss = loss_f + loss_b
                loss.backward()
                if self.max_grad_norm > 0:
                    clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()

                loss_meter.update(loss.item(), x.size(0))

                progress_bar.set_postfix(nll=loss_meter.avg,
                                         lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(x.size(0))
                self.global_step += x.size(0)

        return loss_meter.avg

    def test(self, epoch, net, optimizer, loaders, **kwargs):
        """
        Test step of model returned by model_builder
        """

        device = self.device
        dims = self.data_dims

        net.eval()
        loss_meter = AverageMeter()

        with torch.set_grad_enabled(False):
            with tqdm(total=loaders.dataset.__len__()) as progress_bar:
                for x, condition_params, trajectory in loaders:

                    x = x.to(device)
                    trajectory = trajectory.to(device)
                    # condition_params = condition_params.to(device)

                    results = net(x)
                    # loss = self.forward_loss(x, *results)
                    recon_traj = self.get_y_tracks(results[0])
                    loss = self.forward_loss((x, trajectory), *results, recon_traj)

                    loss_meter.update(loss.item(), x.size(0))

                    progress_bar.set_postfix(nll=loss_meter.avg)
                    progress_bar.update(x.size(0))

                    self.plot_gen_traj(results[0].view(len(x), dims // 2, -1).transpose(1, 2), f"{epoch}a")

        return loss_meter.avg

    def get_y_tracks(self, weights):
        import numpy as np
        num_samples = len(weights)
        condition_params = torch.tensor([np.random.uniform(0, .05, num_samples),
                                         np.random.uniform(0, .05, num_samples),
                                         np.random.uniform(.95, 1., num_samples),
                                         np.random.uniform(.95, 1., num_samples)]).float().T

        dims = weights.size(1)
        dmp = DMP(weights.size()[-1] // 2, dt=1 / 100, d=2, device=self.device)
        y_track, dy_track, ddy_track = dmp.rollout_torch(condition_params[:, :2],
                                                         condition_params[:, -2:],
                                                         weights.view(num_samples, dims // 2, -1).transpose(1, 2),
                                                         device=self.device)
        return y_track

    @staticmethod
    def forward_loss(*args):
        """
        Loss for the labeled data
        """
        raise NotImplementedError("Not implemented labeled loss")

    @staticmethod
    def backward_loss(*args):
        """
        Loss for the unlabeled data
        """
        raise NotImplementedError("Not implemented unlabeled loss")

    def get_datasets(self):
        """
        Returns the dataset that is requested by the dataset param
        """
        train_ds = NavigateFromTo(type='train', data_path=self.input_data)
        valid_ds = NavigateFromTo(type='validate', data_path=self.input_data)

        return train_ds, valid_ds

    def plot_gen_traj(self, weights, epoch):

        import numpy as np
        import matplotlib.pyplot as plt

        condition_params = torch.tensor([np.random.uniform(0, .05, self.num_test_samples),
                                         np.random.uniform(0, .05, self.num_test_samples),
                                         np.random.uniform(.95, 1., self.num_test_samples),
                                         np.random.uniform(.95, 1., self.num_test_samples)]).float().T

        dmp = DMP(self.data_dims // 2, dt=1 / 100, d=2, device=self.device)

        # weights = xs[-1].view(self.num_test_samples, -1, self.data_dims // 2)

        start = condition_params[:len(weights), :2]
        goal = condition_params[:len(weights), -2:]

        dmp.start = start
        dmp.goal = goal
        dmp.y0 = start
        dmp.T = 100

        y_track, dy_track, ddy_track = dmp.rollout_torch(start, goal, weights, device=self.device)

        # ax = plot_trajectories(y_track.detach().numpy(), constraints=trainloader.dataset.constraints)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plot_trajectories(y_track, ax=ax)
        fig.savefig(f'{self.figure_path}/sample_' + str(epoch) + '.png')  # save the figure to file
        plt.close(fig)

    # def get_loaders(self, train_ds, valid_ds):
    #     # import pdb
    #     # pdb.set_trace()
    #     train_loader = torch.utils.data.DataLoader(train_ds, **self.loader_params)
    #     valid_loader = torch.utils.data.DataLoader(valid_ds, **self.loader_params)
    #     return train_loader, valid_loader

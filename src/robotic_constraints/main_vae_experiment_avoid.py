#!/usr/bin/env python

import itertools

from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import torch.nn.utils as utils


# make nflib available
import sys
import os
import logging
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F

from rss_code_and_data import (DMP,gen_weights,imitate_path,plot_rollout)
from robotic_constraints.dataloader import NavigateFromTo

from vaelib.vae import VAE, VAE_Gaussian

from torch.nn.utils import clip_grad_norm_

best_loss = np.inf
global_step = 0
NUM_CATEGORIES = 5
mdl_name = 'vae'

def build_model(data_dim=10, hidden_dim=10, conditioning=False, num_conditioning=4):
    return VAE_Gaussian(data_dim=data_dim, hidden_dim=hidden_dim, condition=conditioning, num_condition=num_conditioning)

def raise_cuda_error():
    raise ValueError('You wanted to use cuda but it is not available. '
                     'Check nvidia-smi and your configuration. If you do '
                         'not want to use cuda, pass the --no-cuda flag.')

def main(args):
    global best_loss

    # Set up main device and scale batch size
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda and not use_cuda:
        raise_cuda_error()

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        logging.info(f'Using device: {torch.cuda.get_device_name()}')

    config_args = [str(vv) for kk, vv in vars(args).items()
                   if kk in ['batch_size', 'lr', 'gamma', "hidden_dim", 'seed', "backward", "back_strength"]]
    model_name = '_'.join(config_args)

    if not os.path.exists(args.output):
        logging.info(f'{args.output} does not exist, creating...')
        os.makedirs(args.output)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Generators
    loader_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
    training_set = NavigateFromTo(type='train', data_path=args.input, trajectory=True)
    trainloader = data.DataLoader(training_set, **loader_params)

    validation_set = NavigateFromTo(type='validate', data_path=args.input, trajectory=True)
    testloader = data.DataLoader(validation_set, **loader_params)

    # Model
    dims = training_set.n_dims
    print(f'Building model with {dims} latent dims')
    net = build_model(dims, hidden_dim=args.hidden_dim)

    print("number of params: ", sum(p.numel() for p in net.parameters()))

    net = net.to(device)

    start_epoch = 0
    save_dir = os.path.join(args.output, 'models')

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at save/best.pth.tar...')
        assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(save_dir, f'{mdl_name}_{model_name}.best.pt'))
        net.load_from_state_dict(checkpoint['net'])
        global global_step
        best_loss = checkpoint['test_loss']
        global_step = start_epoch * len(training_set)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(f'{args.output}/logs/', exist_ok=True)
    log_fh = open(f'{args.output}/logs/{mdl_name}_{model_name}.log', 'w')
    count_valid_not_improving = 0

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = train(epoch, net, trainloader, device, optimizer, scheduler, args.max_grad_norm, args, log_fh)
        vld_loss = test(epoch, net, testloader, device, args.num_samples, save_dir, args, model_name, log_fh)

        scheduler.step()

        if not np.isnan(vld_loss):
            log_fh.write(f'{epoch},{loss},{vld_loss}\n')
            log_fh.flush()

        # early stopping
        if vld_loss >= best_loss:
            count_valid_not_improving += 1

        if count_valid_not_improving > args.early_stopping_lim:
            print(f'Early stopping implemented at epoch #: {epoch}')
            break
        else:
            count_valid_not_improving = 0

        if np.isnan(vld_loss):
            print(f'Early stopping: valid loss is NAN')
            break

    log_fh.close()

    state_curr = { 'net': net.state_dict() }
    torch.save(state_curr, os.path.join(save_dir, f'{mdl_name}_{model_name}.final.pt'))

    # plot_res(net, device, dims)
def forward_loss_traj(y, y_recon, latent_params):
    reconstruction_loss = ((y.view(-1,100,2) - y_recon) ** 2).sum(dim=1)
    mu, logvar = latent_params
    KL_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (reconstruction_loss + KL_term).sum()

def forward_loss(x, reconstruction_x, latent_params):
    # reconstruction loss
    base_dist = MultivariateNormal(torch.zeros_like(x[0]), torch.eye(x.size(1)))
    mu_x, logvar_x = reconstruction_x

    reconstruction_loss = logvar_x + (x - mu_x)**2 / (2*torch.exp(logvar_x))
    # reconstruction_loss = log_var*((x.view(-1,100,2) - reconstruction_x)**2).sum(dim=1)
    # reconstruction_loss = ((x - x_reconstruct) ** 2).sum(dim=1)

    # KL-div term
    mu, logvar = latent_params
    KL_term = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (reconstruction_loss + KL_term).sum()


def backward_loss(y_track, constraints, back_strength=1e3):
    neg_loss = 0
    const = back_strength

    for constraint in constraints:
        neg_loss -= (const * torch.where((y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                         (y_track[:, :, 1] - constraint['coords'][1]) ** 2 < constraint['radius'] ** 2,
                                         (y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
                                         (y_track[:, :, 1] - constraint['coords'][1]) ** 2 - constraint['radius'] ** 2,
                                         torch.zeros_like(y_track[:, :, 0])).sum(dim=1) ** 2)

    neg_loss -= const * torch.where(y_track[:, :, 0] > 1, (y_track[:, :, 0] - 1) ** 2,
                                    torch.zeros_like(y_track[:, :, 0])).sum(dim=1)

    neg_loss -= const * torch.where(y_track[:, :, 1] > 1, (y_track[:, :, 1] - 1) ** 2,
                                    torch.zeros_like(y_track[:, :, 0])).sum(dim=1)

    neg_loss -= const * torch.where(y_track[:, :, 0] < 0, (y_track[:, :, 0]) ** 2,
                                    torch.zeros_like(y_track[:, :, 0])).sum(dim=1)

    neg_loss -= const * torch.where(y_track[:, :, 1] < 0, (y_track[:, :, 1]) ** 2,
                                    torch.zeros_like(y_track[:, :, 0])).sum(dim=1)

    return -neg_loss


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, max_grad_norm, args, log_fh):

    global global_step
    print('\nEpoch: %d' % epoch)

    net.train()
    loss_meter = AverageMeter()

    with tqdm(total=len(trainloader.dataset)) as progress_bar:

        for x, condition_params, trajectory in trainloader:

            x = x.to(device)
            condition_params = condition_params.to(device)
            condition_params_net, _ = net.to_logits(condition_params)

            trajectory = trajectory.to(device)
            dims = x.size(1)

            dmp = DMP(x.size()[-1] // 2, dt=1 / 100, d=2, device=device)

            optimizer.zero_grad()

            # x_reconstructed, latent_params = net(x)
            x_reconstructed, latent_params = net(x, condition_variable=condition_params_net)

            # dist = MultivariateNormal(torch.zeros_like(x_reconstructed[0][0]), torch.eye(len(x_reconstructed[0][0])))
            # xs = x_reconstructed[0] + torch.exp(x_reconstructed[1]) * dist.sample((len(x),))
            #
            # y_reconstructed, _, __ = dmp.rollout_torch(condition_params[:, :2],
            #                                          condition_params[:, -2:],
            #                                          xs.view(len(x), dims // 2, -1).transpose(1, 2),
            #                                          device=device)
            loss = 0
            if args.backward and epoch > 5:
                # optimizer.zero_grad()
                # num_samples = np.min([epoch + 1, len(x)])
                num_samples = len(x)
                # z = net.base.sample((num_samples,))
                # condition_params = torch.tensor([np.random.uniform(0, .1,num_samples),
                #                 np.random.uniform(0,  1.,num_samples),
                #                 np.random.uniform(.9, 1.,num_samples),
                #                 np.random.uniform(0,  1., num_samples)]).float().T.to(device)
                # condition_params_net, _ = net.to_logits(condition_params)

                # xs = net.reconstruct(z, condition_variable=condition_params_net)
                dist = MultivariateNormal(torch.zeros_like(x_reconstructed[0][0]), torch.eye(len(x_reconstructed[0][0])))
                xs = x_reconstructed[0] + torch.exp(x_reconstructed[1])*dist.sample((num_samples, ))

                y_track, dy_track, ddy_track = dmp.rollout_torch(condition_params[:,  :2],
                                                                 condition_params[:, -2:],
                                                                 xs.view(num_samples, dims//2, -1).transpose(1,2),
                                                                 device=device)

                back_loss = backward_loss(y_track, trainloader.dataset.constraints, args.back_strength)
                print(back_loss.sum())
                loss += back_loss.sum()

            # loss += forward_loss_traj(trajectory, y_reconstructed, latent_params)
            loss += forward_loss(x, x_reconstructed, latent_params)
            # loss += latent_losses[0] + latent_losses[1] + latent_losses[2]
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # scheduler.step(global_step
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)

    return loss_meter.avg


@torch.no_grad()
def test(epoch, net, testloader, device, num_samples, save_dir, args, model_name, log_fh):
    global best_loss
    net.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, condition_params, trajectory in testloader:
            x = x.to(device)
            condition_params = condition_params.to(device)
            condition_params_net, _ = net.to_logits(condition_params)

            trajectory = trajectory.to(device)
            dims = x.size(1)

            dmp = DMP(x.size()[-1] // 2, dt=1 / 100, d=2, device=device)

            # x_reconstructed, latent_params = net(x)
            x_reconstructed, latent_params = net(x, condition_variable=condition_params_net)

            # dist = MultivariateNormal(torch.zeros_like(x_reconstructed[0][0]), torch.eye(len(x_reconstructed[0][0])))
            # xs = x_reconstructed[0] + torch.exp(x_reconstructed[1]) * dist.sample((len(x),))
            #
            # y_reconstructed, _, __ = dmp.rollout_torch(condition_params[:, :2],
            #                                            condition_params[:, -2:],
            #                                            xs.view(len(x), dims // 2, -1).transpose(1, 2),
            #                                            device=device)
            # y_reconstructed, _, __ = dmp.rollout_torch(condition_params[:, :2],
            #                                            condition_params[:, -2:],
            #                                            x_reconstructed.view(x.size(0), -1, dims//2),
            #                                            device=device)
            # loss = forward_loss_traj(trajectory, y_reconstructed, latent_params)
            # loss = forward_loss(prior_logprob, log_det)
            # loss = forward_loss(trajectory, y_reconstructed, latent_params)
            loss = forward_loss(x, x_reconstructed, latent_params)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg)
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
    # if True:
        print(f'Saving...  {mdl_name}_{model_name}.best.pt')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state, os.path.join(save_dir, f'{mdl_name}_{model_name}.best.pt'))
        best_loss = loss_meter.avg

    # Save samples and data
    # images = sample(net, num_samples, device)
    # samp_dir = os.path.join(save_dir, 'save')
    # os.makedirs(samp_dir, exist_ok=True)
    # images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    # torchvision.utils.save_image(images_concat,
    #                              os.path.join(samp_dir, 'epoch_{}.png'.format(epoch)))

    return loss_meter.avg

def plot_res(net, device, dims):
    from robotic_constraints.utils import plot_trajectories
    import matplotlib.pyplot as plt
    num_samples = 1000
    NUM_CATEGORIES = 5

    z = net.prior.sample(num_samples)
    condition_params = torch.tensor([
        np.random.uniform(0, .1, num_samples),
        np.random.uniform(0, 1., num_samples),
        np.random.uniform(.9, 1., num_samples),
        np.random.uniform(0.499, .501, num_samples)]).float().T

    labels = torch.zeros(num_samples, NUM_CATEGORIES).to(device)
    labels[torch.arange(num_samples), torch.arange(NUM_CATEGORIES).repeat(num_samples // NUM_CATEGORIES)] = 1

    # run the flow
    xs, logdet_backward = net.backward(labels, condition_variable=condition_params)
    weights = xs[-1].view(num_samples, dims // 2, -1).transpose(1, 2)
    dmp = DMP(dims // 2, dt=1 / 100, d=2, device=device)

    start = condition_params[:len(weights), :2]
    goal = condition_params[:len(weights), 2:]

    dmp.start = start
    dmp.goal = goal
    dmp.y0 = start
    dmp.T = 100

    y_track, dy_track, ddy_track = dmp.rollout_torch(start, goal, weights, device=device)
    ax = plot_trajectories(y_track.detach().numpy())
    plt.show()



class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def construct_parser():
    parser = argparse.ArgumentParser(description='Flow++ on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=False, help='Turn on CUDNN benchmarking')
    parser.add_argument('--backward', type=str2bool, default=False, help='Turn on semantic loss for backward eval')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='Turn on CUDA')
    parser.add_argument('--lr', default=1e-3, type=float, help='Peak learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--hidden_dim', default=20, type=int, help='Number of hidden dimensions')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_dequant_blocks', default=0, type=int, help='Number of blocks in dequantization')
    parser.add_argument('--back_strength', type=float, default=1e3, help='Strength of backward loss')
    parser.add_argument('--num_channels', default=96, type=int, help='Number of channels in Flow++')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--use_attn', type=str2bool, default=True, help='Use attention in the coupling layers')
    parser.add_argument('--save_dir', type=str, default='samples', help='Directory for saving samples')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input data dir')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output data dir')
    parser.add_argument('--warm_up', type=int, default=200, help='Number of batches for LR warmup')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--early-stopping-lim', type=int, default=250, metavar='N',
                        help='Early stopping implemented after N epochs with no improvement '
                             '(default: 10)')
    return parser


if __name__ == '__main__':
    print("execute the main script in ../main_experiment_avoid.py")

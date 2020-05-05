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
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from tqdm import tqdm
#
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from nflib import (AffineConstantFlow,
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
    BaseDistribution, BaseDistributionMixtureGaussians,
                   SS_Flow
)

from rss_code_and_data import (DMP,gen_weights,imitate_path,plot_rollout)


# from nflib.utils import (plot_model, ConstrainedGaussian, DatasetMoons, WeirdConstrainedGaussian,
#                          DatasetSIGGRAPH, DatasetMixture, ConstrainedGaussianInner)
# import rss_code_and_data.code.curve_funcs as cf
from robotic_constraints.dataloader import NavigateFromTo


best_loss = 0
global_step = 0
NUM_CATEGORIES = 5

mdl_name = 'maf'

def build_model(dim=10, num_layers=10, conditioning=True, num_conditioning=4):

    # base_dist = BaseDistribution(dim)
    base_dist = BaseDistributionMixtureGaussians(dim, NUM_CATEGORIES)
    #
    flows = [MAF(dim=dim, parity=i%2, conditioning=conditioning, num_conditioning=num_conditioning)
             for i in range(num_layers)]
    convs = [Invertible1x1Conv(dim=dim) for _ in flows]
    norms = [ActNorm(dim=dim) for _ in flows]

    flows = list(itertools.chain(*zip(flows, convs, norms)))

    return NormalizingFlowModel(base_dist, flows, conditioning=conditioning)


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
                   if kk in ['batch_size', 'lr', 'gamma', "num_layers", 'seed', "backward"]]
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
    training_set = NavigateFromTo(type='train', data_path=args.input)
    trainloader = data.DataLoader(training_set, **loader_params)

    validation_set = NavigateFromTo(type='validate', data_path=args.input)
    testloader = data.DataLoader(validation_set, **loader_params)

    # Model
    dims = training_set.n_dims
    print(f'Building model with {dims} latent dims')
    net = SS_Flow(flows=build_model(dim=dims, num_layers=args.num_layers, conditioning=False, num_conditioning=0),
                  NUM_CATEGORIES=NUM_CATEGORIES, dims=dims)
    # net = build_model(dim=dims, num_layers=args.num_layers, conditioning=False, num_conditioning=4)
    # net = FlowPlusPlus(scales=[(0, 4), (2, 3)],
    #                    in_shape=(1, 1, 50),
    #                    mid_channels=args.num_channels,
    #                    num_blocks=args.num_blocks,
    #                    num_dequant_blocks=0,
    #                    num_components=args.num_components,
    #                    use_attn=args.use_attn,
    #                    drop_prob=args.drop_prob)
    print("number of params: ", sum(p.numel() for p in net.parameters()))

    net = net.to(device)

    # if device == 'cuda':
        # net = torch.nn.DataParallel(net, args.gpu_ids)
        # cudnn.benchmark = args.benchmark

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
        # print(best_loss)
        # start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(training_set)

    # args.lr = 1e-8

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(f'{args.output}/logs/', exist_ok=True)
    log_fh = open(f'{args.output}/logs/{mdl_name}_{model_name}.log', 'w')
    count_valid_not_improving = 0

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = train(epoch, net, trainloader, device, optimizer, scheduler, args.max_grad_norm, args, log_fh)
        vld_loss = test(epoch, net, testloader, device, args.num_samples, save_dir, args, model_name, log_fh)

        net.tau = np.max((0.5, net.tau * np.exp(-5e-3 * (epoch))))
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

    state_curr = { 'net': net.get_state_params() }
    torch.save(state_curr, os.path.join(save_dir, f'{mdl_name}_{model_name}.final.pt'))

    plot_res(net, device, dims)

def forward_loss(prior_logprob, log_det):
    logprob = prior_logprob + log_det
    return torch.sum(logprob)


def backward_loss(neg_loss, log_det):
    return -(torch.sum(neg_loss) + log_det.sum())
    # return -torch.sum(neg_loss)


def combined_loss(prior_logprob, log_det_fwd, neg_loss, log_det_bkwd):
    return forward_loss(prior_logprob, log_det_fwd) + backward_loss(neg_loss, log_det_bkwd)


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, max_grad_norm, args, log_fh):

    global global_step
    print('\nEpoch: %d' % epoch)

    # import pdb
    # pdb.set_trace()

    # for x, _ in enumerate(trainloader):
        # print(x)

    net.train()
    loss_meter = AverageMeter()

    with tqdm(total=len(trainloader.dataset)) as progress_bar:

        for x, condition_params, _ in trainloader:

            x = x.to(device)
            condition_params = condition_params.to(device)

            optimizer.zero_grad()

            zs, latent_losses, pred_label = net((x,None), condition_variable=condition_params)
            # zs, prior_logprob, log_det = net(x)
            prior_logprob, log_det, cat_kl = latent_losses

            loss = 0
            if args.backward and epoch > 10:
                # optimizer.zero_grad()
                # regularizer = MultivariateNormal(torch.zeros_like(x[-1]).to(device),
                #                                  .25 * torch.eye(x.size()[1]).to(device))

                # num_samples = np.min([(epoch+2), len(x)//10])
                num_samples = len(x)
                # z = net.prior.sample(num_samples)
                condition_params = torch.tensor([np.random.uniform(0, .05, num_samples),
                                                 np.random.uniform(0, .05, num_samples),
                                                 np.random.uniform(.95, 1., num_samples),
                                                 np.random.uniform(.95, 1., num_samples)]).float().T

                labels = torch.zeros(num_samples, NUM_CATEGORIES).to(device)
                labels[torch.arange(num_samples), torch.arange(NUM_CATEGORIES).repeat(num_samples // NUM_CATEGORIES)] = 1

                xs, log_det_back = net.backward(labels, condition_variable=condition_params)
                # zs, prior_logprob, log_det = net.backward(z)

                dims = x.size(1)
                dmp = DMP(x.size()[-1]//2, dt=1 / 100, d=2, device=device)
                y_track, dy_track, ddy_track = dmp.rollout_torch(condition_params[:,  :2],
                                                                 condition_params[:, -2:],
                                                                 xs.view(num_samples, dims//2, -1).transpose(1,2),
                                                                 device=device)

                # neg_loss = regularizer.log_prob(xs)
                neg_loss = 0
                const = 1e2
                for constraint in trainloader.dataset.constraints:
                    neg_loss -= (const*torch.where((y_track[:, :, 0] - constraint['coords'][0]) ** 2 +
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

                back_loss = backward_loss(neg_loss, log_det_back[neg_loss<0])
                # back_loss = -neg_loss.sum()
                # if neg_loss.sum() > 0:
                print(neg_loss.sum().item(), back_loss.item())
                # back_loss.backward()
                # if max_grad_norm > 0:
                #     clip_grad_norm(optimizer, max_grad_norm)
                # optimizer.step()
                loss += back_loss

            loss += forward_loss(prior_logprob, (log_det+cat_kl))
            # loss += latent_losses[0] + latent_losses[1] + latent_losses[2]
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            # scheduler.step(global_step
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)

    return loss_meter.avg


@torch.no_grad()
def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
def test(epoch, net, testloader, device, num_samples, save_dir, args, model_name, log_fh):
    global best_loss
    net.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, condition_params, _ in testloader:
            x = x.to(device)
            condition_params = condition_params.to(device)

            # zs, prior_logprob, log_det = net(x)
            zs, latent_losses, pred_label = net((x, None), condition_variable=condition_params)
            prior_logprob, log_det, cat_kl = latent_losses
            loss = forward_loss(prior_logprob, (log_det + cat_kl))
            # loss = latent_losses[0] + latent_losses[1] + latent_losses[2]
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg)
            progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
    # if True:
        print(f'Saving...  {mdl_name}_{model_name}.best.pt')
        state = {
            'net': net.get_state_params(),
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
    condition_params = torch.tensor([np.random.uniform(0, .05, num_samples),
                                     np.random.uniform(0, .05, num_samples),
                                     np.random.uniform(.95, 1., num_samples),
                                     np.random.uniform(.95, 1., num_samples)]).float().T

    labels = torch.zeros(num_samples, NUM_CATEGORIES).to(device)
    labels[torch.arange(num_samples), torch.arange(NUM_CATEGORIES).repeat(num_samples // NUM_CATEGORIES)] = 1

    # run the flow
    zs, latent_losses, pred_label = net.backward(labels, condition_variable=condition_params)
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
    parser.add_argument('--num_layers', default=20, type=int, help='Number of layers in Flow')
    parser.add_argument('--num_components', default=32, type=int, help='Number of components in the mixture')
    parser.add_argument('--num_blocks', default=10, type=int, help='Number of blocks in Flow++')
    parser.add_argument('--num_dequant_blocks', default=0, type=int, help='Number of blocks in dequantization')
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

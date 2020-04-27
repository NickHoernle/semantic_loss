from __future__ import print_function
import argparse
import torch
import os
import logging
import random
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal, Uniform, \
    TransformedDistribution, SigmoidTransform, Normal, Categorical

from vaelib.vae import VAE, VAE_Categorical
from torch.nn.utils import clip_grad_norm_

from utils.logging_utils import AverageMeter, raise_cuda_error
from utils.data import get_samplers

best_loss = np.inf
global_step = 0
NUM_CATEGORIES = 5
mdl_name = 'vae'

def build_model(data_dim=10, hidden_dim=10, conditioning=True, num_conditioning=4):
    return VAE(data_dim=data_dim, hidden_dim=hidden_dim, condition=conditioning, num_condition=num_conditioning)


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
    loader_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': args.num_workers}
    if args.dataset == 'MNIST':
        train_ds = datasets.MNIST(args.input, train=True, download=True, transform=transforms.ToTensor())
        test_ds = datasets.MNIST(args.input, train=False, transform=transforms.ToTensor())
        NUM_CATEGORIES=10
    elif args.dataset == 'CIFAR10':
        train_ds = datasets.MNIST(args.input, train=True, download=True, transform=transforms.ToTensor())
        test_ds = datasets.MNIST(args.input, train=False, transform=transforms.ToTensor())
        NUM_CATEGORIES=10
    elif args.dataset == 'CIFAR100':
        train_ds = datasets.MNIST(args.input, train=True, download=True, transform=transforms.ToTensor())
        test_ds = datasets.MNIST(args.input, train=False, transform=transforms.ToTensor())
        NUM_CATEGORIES=100
    else:
        raise ValueError("Dataset not in {MNIST|CIFAR10|CIFAR100}")

    labelled_sampler, unlabelled_sampler = get_samplers(train_ds.train_labels.numpy(), n=100, n_categories=NUM_CATEGORIES)
    train_loader_labelled = torch.utils.data.DataLoader(train_ds, sampler=labelled_sampler, **loader_params)
    train_loader = torch.utils.data.DataLoader(train_ds, sampler=unlabelled_sampler, **loader_params)
    test_loader = torch.utils.data.DataLoader(test_ds, **loader_params)

    # length = 0
    # for data_u, labels_u in train_loader:
    #     length += data_u.size(0)
    #
    # print(length)

    # Model
    dims = 1
    for i in train_loader.dataset.__getitem__(0)[0].shape:
        dims *= i

    print(f'Building model with {dims} latent dims')
    net = VAE_Categorical(data_dim=dims, hidden_dim=args.hidden_dim, NUM_CATEGORIES=NUM_CATEGORIES)

    print("number of params: ", sum(p.numel() for p in net.parameters()))
    net = net.to(device)

    start_epoch = 0
    save_dir = os.path.join(args.output, 'models')

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at save/best.pth.tar...')
        assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(save_dir, f'{mdl_name}_{model_name}.best.pt'))
        net.load_state_dict(checkpoint['net'])
        global global_step
        best_loss = checkpoint['test_loss']
        global_step = start_epoch * len(train_loader.dataset)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    os.makedirs(f'{args.output}/logs/', exist_ok=True)
    log_fh = open(f'{args.output}/logs/{mdl_name}_{model_name}.log', 'w')
    count_valid_not_improving = 0

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        loss = train(epoch, net, (train_loader_labelled, train_loader), device, optimizer, scheduler, args.max_grad_norm, args, dims)
        vld_loss = test(epoch, net, test_loader, device, args.num_samples, save_dir, args, model_name, dims)

        scheduler.step()

        net.tau = np.max((0.5, net.tau * np.exp(-5e-3 * (epoch))))

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

        with torch.no_grad():

            labels = torch.zeros(64, NUM_CATEGORIES).to(device)
            labels[torch.arange(64), torch.arange(8).repeat(8)] = 1

            img_sample = net.sample_labelled(labels)

            img_sample = torch.sigmoid(img_sample)
            save_image(img_sample.view(64, 1, 28, 28), '../experiments/vaelib/samples/sample_' + str(epoch) + '.png')

    log_fh.close()

    state_curr = {'net': net.state_dict()}
    torch.save(state_curr, os.path.join(save_dir, f'{mdl_name}_{model_name}.final.pt'))

def labelled_loss(data, data_reconstructed, latent_params, label_sample):
    BCE = F.binary_cross_entropy(torch.sigmoid(data_reconstructed), data, reduction='sum')
    mu, logvar, label_logprob = latent_params

    KLD_continuous = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    discriminator_loss = -(label_sample*(label_logprob)).sum(dim=1).sum()

    return BCE + KLD_continuous + discriminator_loss


def unlabelled_loss(data, data_reconstructed, latent_params, label_sample):
    BCE = F.binary_cross_entropy(torch.sigmoid(data_reconstructed), data, reduction='sum')
    mu, logvar, label_logprob, prior = latent_params

    KLD_continuous = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD_discrete = -(label_sample * (label_logprob+prior)).sum(dim=1).sum()

    return BCE + KLD_continuous + KLD_discrete

@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, max_grad_norm, args, dims):

    global global_step
    print('\nEpoch: %d' % epoch)

    net.train()
    loss_meter = AverageMeter()

    (train_loader_labelled_, train_loader) = trainloader
    train_loader_labelled = iter(train_loader_labelled_)

    with tqdm(total=len(train_loader.dataset)) as progress_bar:

        for data_u, labels_u in train_loader:

            try:
                data_l, target_l = next(train_loader_labelled)
            except StopIteration:
                train_loader_labelled = iter(train_loader_labelled_)
                data_l, target_l = next(train_loader_labelled)

            data_u = data_u.view(-1,dims).to(device)

            data_l = data_l.view(-1, dims).to(device)
            one_hot = VAE_Categorical.convert_to_one_hot(
                num_categories=net.NUM_CATEGORIES,
                labels=target_l).to(device)

            optimizer.zero_grad()

            data_reconstructed_l, latent_params_l, label_sample_l = net((data_l, one_hot))
            data_reconstructed_u, latent_params_u, label_sample_u = net((data_u, None))

            loss_l = labelled_loss(data_l, data_reconstructed_l, latent_params_l, one_hot)
            loss_u = unlabelled_loss(data_u, data_reconstructed_u, latent_params_u, label_sample_u)

            loss = loss_l + loss_u

            loss_meter.update(loss.item(), data_u.size(0))
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(data_u.size(0))
            global_step += data_u.size(0)

    return loss_meter.avg


@torch.no_grad()
def test(epoch, net, testloader, device, num_samples, save_dir, args, model_name, dims):
    global best_loss
    net.eval()
    loss_meter = AverageMeter()
    accuracy = []

    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for data, labels in testloader:
            data = data.view(-1, dims).to(device)
            one_hot = VAE_Categorical.convert_to_one_hot(
                num_categories=net.NUM_CATEGORIES,
                labels=labels).to(device)

            data_reconstructed, latent_params, label_sample = net((data, one_hot))

            loss = labelled_loss(data, data_reconstructed, latent_params, label_sample)
            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg)
            progress_bar.update(data.size(0))

            accuracy += ((torch.argmax(latent_params[-1], dim=1) == labels).float()
                     .detach().numpy()).tolist()

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print(f'Saving...  {mdl_name}_{model_name}.best.pt')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state, os.path.join(save_dir, f'{mdl_name}_{model_name}.best.pt'))
        best_loss = loss_meter.avg

    print(f'===============> Accuracy: {np.mean(accuracy)}')

    return loss_meter.avg


def construct_parser():
    parser = argparse.ArgumentParser(description='Categorical VAE')

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
    parser.add_argument('--dataset', default='MNIST', type=str, help='Dataset. One of MNIST|CIFAR10|CIFAR100')
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
    print("execute the main script in ../main_vae_categorical.py")
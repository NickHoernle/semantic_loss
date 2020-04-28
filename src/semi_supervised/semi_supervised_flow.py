"""
Author: Nick Hoernle
Base class that defines the main, train, test methods that are used by the ML models in this repository. 
Each experimet will extend this class to define it's own parameters for the experiment but this class allows 
for more efficiency in the code.
"""

from __future__ import print_function

import os
import random
import logging

import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms


from utils.logging import raise_cuda_error, AverageMeter
from utils.generative_trainer import GenerativeTrainer

from nflib import (AffineConstantFlow,
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
    BaseDistribution, BaseDistributionMixtureGaussians,
    SS_Flow
)


def build_model(dim=10, num_layers=10, conditioning=True, num_conditioning=10):

    base_dist = BaseDistributionMixtureGaussians(dim, NUM_CATEGORIES)

    flows = [MAF(dim=dim, parity=i%2, conditioning=conditioning, num_conditioning=num_conditioning, nh=25)
             for i in range(num_layers)]
    convs = [Invertible1x1Conv(dim=dim) for _ in flows]
    norms = [ActNorm(dim=dim) for _ in flows]

    flows = list(itertools.chain(*zip(flows, convs, norms)))

    return NormalizingFlowModel(base_dist, flows, conditioning=conditioning)

class CategoricalFlowGenerativeTrainer(GenerativeTrainer):
    def __init__(self, model_builder, model_parameters, input_data, output_data, dataset='MNIST', max_grad_norm=1, num_epochs=100, batch_size=256, lr=1e-3, use_cuda=True, num_test_samples=256, seed=0, gamma=0.9, resume=False, early_stopping_lim=50, additional_model_config_args=[], num_loader_workers=8, data_shuffle=True, name="base-model",):

        train_ds, valid_ds, NUM_CATEGORIES = get_dataset(input_data, dataset)
        super().__init__(model_builder, model_parameters, (train_ds, valid_ds), output_data, max_grad_norm=1, num_epochs=100, batch_size=256, lr=1e-3, use_cuda=True, num_test_samples=256, seed=0, gamma=0.9, resume=False, early_stopping_lim=50, additional_model_config_args=[], num_loader_workers=8, data_shuffle=True, name="base-model",)

        self.dataset = dataset # Must be one of MNIST|CIFAR10|CIFAR100'
        self.NUM_CATEGORIES = NUM_CATEGORIES

    @torch.enable_grad()
    def train(self, epoch, optimizer, **kwargs):
        """
        Train step of model returned by model_builder
        """
        net = self.net

        net.train()
        loss_meter = AverageMeter()

        (train_loader_labelled_, train_loader) = trainloader
        train_loader_labelled = iter(train_loader_labelled_)

    @torch.no_grad()
    def test(self, epoch, optimizer, **kwargs):
        """
        Test step of model returned by model_builder
        """
        raise NotImplementedError(
            "Class must be overridden and test should be implemented"
        )


def get_dataset(input_data, dataset):
    if dataset == 'MNIST':
        train_ds = datasets.MNIST(input_data, train=True, download=True, transform=transforms.ToTensor())
        valid_ds = datasets.MNIST(input_data, train=False, transform=transforms.ToTensor())
        num_categories = 10
    elif dataset == 'CIFAR10':
        train_ds = datasets.MNIST(input_data, train=True, download=True, transform=transforms.ToTensor())
        valid_ds = datasets.MNIST(input_data, train=False, transform=transforms.ToTensor())
        num_categories = 10
    elif dataset == 'CIFAR100':
        train_ds = datasets.MNIST(input_data, train=True, download=True, transform=transforms.ToTensor())
        valid_ds = datasets.MNIST(input_data, train=False, transform=transforms.ToTensor())
        num_categories = 100
    else:
        raise ValueError("Dataset not in {MNIST|CIFAR10|CIFAR100}")
    return train_ds, valid_ds, num_categories


def labelled_loss(data, data_reconstructed, latent_params, label_sample):
    #TODO
    return 0


def unlabelled_loss(data, data_reconstructed, latent_params, label_sample):
    #TODO
    return 0


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
            one_hot = convert_to_one_hot(
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









kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1000, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=True, **kwargs)

device = torch.device("cpu")
NUM_CATEGORIES = 10

from torch.nn.utils import clip_grad_norm_


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight,0,.05)
        m.bias.data.fill_(0)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


model = SS_Flow(flows=build_model(dim=28 * 28, num_layers=5, conditioning=False, num_conditioning=0),
                NUM_CATEGORIES=NUM_CATEGORIES).to(device)

print("number of params: ", sum(p.numel() for p in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)


def train(epoch, args):

    model.train()
    train_loss = 0

    model.tau = np.max((0.5, model.tau * np.exp(-5e-3 * (epoch))))

    print(model.tau, scheduler.get_lr()[0])

    unlab, lab = 0, 0

    for batch_idx, (data, labels) in enumerate(train_loader):

        if batch_idx == 0:
            labeled_data, labeled = data, labels
            continue
        optimizer.zero_grad()

        recon_batch_l, latent_losses_l, pred_labels_l = model((labeled_data, labeled))
        loss = - (latent_losses_l[0] + latent_losses_l[1]) + latent_losses_l[2]


        if batch_idx > 2:
            recon_batch, latent_losses, pred_labels = model((data, None))
            loss += latent_losses[0] + latent_losses[1] + latent_losses[2]

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        train_loss += loss.item()

        optimizer.step()
        scheduler.step()

        # update the distribution parameters
        # model.flow_main.prior.set_means_from_latent_data(recon_batch_l, pred_labels_l)
        # if epoch > 10:
        # model.flow_main.prior.set_means_from_latent_data_known_labels(recon_batch_l, labels)

        if (batch_idx % args.log_interval == 0) and batch_idx > 2:

            labels = torch.unsqueeze(labels, 1)
            one_hot = torch.FloatTensor(labels.size()[0], NUM_CATEGORIES).zero_()
            one_hot.scatter_(1, labels, 1)

            pred_loss = -(one_hot*(pred_labels_l)).sum(dim=1).sum()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLab Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                pred_loss))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    print(f"\t\t {unlab}/{unlab+lab}")

def test(epoch, *args):
    model.eval()
    test_loss = 0

    pred_loss = 0
    accuracy = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):

            data = data.to(device)
            labels = labels.to(device)

            recon_batch, latent_losses, pred_labels = model((data, labels))
            test_loss += latent_losses[0] + latent_losses[1] #+ latent_losses[2]

            labels = torch.unsqueeze(labels, 1)
            one_hot = torch.FloatTensor(labels.size()[0], NUM_CATEGORIES).zero_()
            one_hot.scatter_(1, labels, 1)

            pred_loss += -(one_hot * pred_labels).sum(dim=1).sum()

            accuracy += ((torch.argmax(torch.softmax(pred_labels, dim=1), dim=1) == labels[:,0]).float()
                         .detach().numpy()).tolist()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    pred_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('=========> Pred set loss: {:.4f}'.format(pred_loss))
    print(f'===============> Accuracy: {np.mean(accuracy)}')


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        train(epoch, args)
        test(epoch, args)
        with torch.no_grad():

            labels = torch.zeros(64, NUM_CATEGORIES).to(device)
            labels[torch.arange(64), torch.arange(8).repeat(8)] = 1

            img_sample = model.backward(labels)

            # undoing the logits
            # bounds = torch.tensor([0.9], dtype=torch.float32)
            img_sample = torch.sigmoid(img_sample)
            bounds = torch.tensor([0.9], dtype=torch.float32)
            img_sample = img_sample*2 - 1
            img_sample = (img_sample/bounds + 1)/2
            img_sample[img_sample > 1] = 1
            img_sample[img_sample < 0] = 0

            # if epoch > 5:
            #     import pdb
            #     pdb.set_trace()
            # img_sample = img_sample*2 - 1
            # img_sample = (img_sample/bounds + 1)/2

            # reverse dequant / logits
            # img_sample = torch.sigmoid(img_sample)

            # import pdb
            # pdb.set_trace()

            # img_sample = img_sample-(img_sample.min(dim=1)[0].view(-1,1))
            # img_sample = img_sample/(img_sample.max(dim=1)[0].view(-1,1))

            save_image(img_sample.view(64, 1, 28, 28), '../experiments/vaelib/sample_' + str(epoch) + '.png')

def construct_parser():

    def str2bool(s):
        return s.lower().startswith('t')

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser


if __name__ == "__main__":
    main()
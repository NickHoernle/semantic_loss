from __future__ import print_function
import argparse
import torch
import itertools
import numpy as np

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal, Uniform, \
    TransformedDistribution, SigmoidTransform, Normal, Categorical

from nflib import (AffineConstantFlow,
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
    BaseDistribution, BaseDistributionMixtureGaussians,
    SS_Flow
)

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


def build_model(dim=10, num_layers=10, conditioning=True, num_conditioning=10):

    base_dist = BaseDistributionMixtureGaussians(dim, NUM_CATEGORIES)

    flows = [MAF(dim=dim, parity=i%2, conditioning=conditioning, num_conditioning=num_conditioning, nh=25)
             for i in range(num_layers)]
    convs = [Invertible1x1Conv(dim=dim) for _ in flows]
    norms = [ActNorm(dim=dim) for _ in flows]

    flows = list(itertools.chain(*zip(flows, convs, norms))) # append a coupling layer after each 1x1

    # flows.append(AffineConstantFlow(dim=dim, shift=False))
    # convs = [Invertible1x1Conv(dim=dim) for _ in flows]
    # norms = [ActNorm(dim=dim) for _ in flows]
    #
    # flows = list(itertools.chain(*zip(norms, convs, flows)))

    return NormalizingFlowModel(base_dist, flows, conditioning=conditioning)

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
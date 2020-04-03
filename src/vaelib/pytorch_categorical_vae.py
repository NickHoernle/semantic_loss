from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal, Uniform, \
    TransformedDistribution, SigmoidTransform, Normal, Categorical

from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
    NormalizingFlowModelWithCategorical
)
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
NUM_CATEGORIES = 10
HIDDEN_DIM = 50
nce_loss = nn.NLLLoss(reduction='sum')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight,0,.05)
        m.bias.data.fill_(0)

def build_flows(num_layers=3, dim=20, conditioning=True, num_conditioning=NUM_CATEGORIES):

    base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    flows = [AffineHalfFlow(
                        dim=dim,
                        parity=i%2,
                        conditioning=conditioning,
                        nh=12,
                        num_conditioning=num_conditioning
                    ) for i in range(num_layers)]

    return NormalizingFlowModel(base_dist, flows, conditioning=conditioning)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

def gumbel_softmax_logpdf(logits, sample, tau=.5):
    '''
    Implementation of the gumbel-softmax PDF
    '''
    k = torch.tensor(sample.size()[1]).float()
    return (k-1)*torch.log(torch.tensor(tau).float()) \
           - k*torch.logsumexp(torch.log(torch.exp(logits)/(sample**tau)), dim=1) \
           + (torch.log(torch.exp(logits)/(sample**(tau+1)))).sum(dim=1)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encoding layers
        self.enc1 = nn.Linear(784, 500)

        # discriminator
        self.disc1 = nn.Linear(500, 250)
        self.disc2 = nn.Linear(250, NUM_CATEGORIES)

        # enc2
        self.enc2 = nn.Linear(500+NUM_CATEGORIES, 250)
        self.enc2_mu = nn.Linear(250, HIDDEN_DIM)
        self.enc2_sig = nn.Linear(250, HIDDEN_DIM)

        # decoding layers
        self.fc3 = nn.Linear(HIDDEN_DIM+NUM_CATEGORIES, 250)
        self.fc3a = nn.Linear(250, 500)
        self.fc4 = nn.Linear(500, 784)

        # flow layers
        # self.flow = build_flows(num_layers=6, dim=HIDDEN_DIM, num_conditioning=NUM_CATEGORIES)

        # self.apply(init_weights)

        # activations
        self.relu = nn.LeakyReLU(.01)
        self.tau = 2

        self.apply(init_weights)

    def encode(self, x):

        h1 = self.relu(self.enc1(x))

        h2 = self.relu(self.disc1(h1))
        log_prob = self.disc2(h2)

        return h1, h2, log_prob

    def reparameterize(self, mu, logvar):

        sigma = torch.exp(logvar)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma

        return z

    def decode(self, latent_samp, one_hot):
        h0 = torch.cat((latent_samp, one_hot), dim=1)
        h3 = self.relu(self.fc3(h0))
        h3a = self.relu(self.fc3a(h3))
        return torch.sigmoid(self.fc4(h3a))

    def get_latent_sample(self, z, mu, logvar):

        # equation 6 in Kingma 2014
        # log p(y), logp(z) - log q(y,z | x)
        q = Normal(mu, torch.exp(logvar))
        std_norm = Normal(torch.zeros_like(z), torch.ones_like(z))

        latent_loss = std_norm.log_prob(z).sum(dim=1) - q.log_prob(z).sum(dim=1)

        return z, latent_loss

    def forward(self, sample):
        (data, labels) = sample
        data = data.to(device)
        if type(labels) != type(None):
            labels = labels.to(device)
            return self.forward_labelled(data, labels)
        return self.forward_unlabelled(data)

    def forward_unlabelled(self, x):
        enc_hidden, sig_hidden, log_pred_label = self.encode(x.view(-1, 784))

        # prior
        log_pis = -torch.log(torch.tensor(NUM_CATEGORIES).float().to(device))
        # sample label
        sampled_labels = F.gumbel_softmax(log_pred_label, tau=self.tau, hard=False)
        log_pred_label_sm = torch.log(torch.softmax(log_pred_label, dim=1))

        # print(log_pred_label_sm)

        # pred_ll = gumbel_softmax_logpdf(log_pred_label, sampled_labels, tau=self.tau)
        # pred_ll = (log_pred_label*sampled_labels).sum(dim=1)
        pred_ll = -(sampled_labels * (log_pred_label_sm - torch.log(torch.tensor(1.0 / NUM_CATEGORIES)))).sum(dim=1)
        # pred_ll = (sampled_labels * (log_pred_label_sm)).sum(dim=1)
        # print(pred_ll)
        # pred_ll = 0

        enc_hidden = self.relu(self.enc2(torch.cat((enc_hidden, sampled_labels), dim=1)))
        mu = self.enc2_mu(enc_hidden)
        logvar = self.enc2_sig(sig_hidden)

        # mu = self.fc21(mu_prime)
        z = self.reparameterize(mu, logvar)
        samp, latent_loss = self.get_latent_sample(z, mu, logvar)

        return self.decode(samp, sampled_labels), -(latent_loss+pred_ll).sum(), log_pred_label_sm

    def forward_labelled(self, x, y):

        # convert labels to one hot
        labels = torch.unsqueeze(y, 1)
        one_hot = torch.FloatTensor(x.size()[0], NUM_CATEGORIES).zero_()
        one_hot.scatter_(1, labels, 1)

        enc_hidden, sig_hidden, log_pred_label = self.encode(x.view(-1, 784))
        enc_hidden = self.relu(self.enc2(torch.cat((enc_hidden, one_hot), dim=1)))
        mu = self.enc2_mu(enc_hidden)
        logvar = self.enc2_sig(sig_hidden)

        # prior
        log_pis = -torch.log(torch.tensor(NUM_CATEGORIES).float().to(device))
        # predicted label
        # log_pred_label += log_pis

        # sample label
        # sampled_labels = F.gumbel_softmax(log_pred_label, tau=.5, hard=False)
        log_pred_label_sm = torch.log(torch.softmax(log_pred_label, dim=1))

        # mu = self.fc21(mu_prime)
        z = self.reparameterize(mu, logvar)
        samp, latent_loss = self.get_latent_sample(z, mu, logvar)

        pred_loss = -(one_hot*(log_pred_label_sm)).sum(dim=1)
        # pred_loss = 0

        return self.decode(samp, one_hot), -latent_loss.sum()+0.1*pred_loss.sum(), log_pred_label_sm


model = VAE().to(device)
print("number of params: ", sum(p.numel() for p in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, latent_loss, pred_labels, true_labels):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # nce_loss_val = nce_loss(torch.log_softmax(pred_labels, dim=1), true_labels).sum()

    return BCE + latent_loss #+ 0.1*nce_loss_val


def train(epoch):
    model.train()
    train_loss = 0
    import numpy as np

    model.tau = np.max((0.5, model.tau * np.exp(-5e-3 * (epoch))))
    # if epoch < 25:
    #     model.tau = 2
    print(model.tau, scheduler.get_lr()[0])

    unlab, lab = 0, 0

    for batch_idx, (data, labels) in enumerate(train_loader):

        if batch_idx == 0:
            labeled_data, labeled = data, labels
            continue

        recon_batch, latent_loss, pred_labels = model((data, None))
        loss = loss_function(recon_batch, data, latent_loss, pred_labels, labels)

        recon_batch_l, latent_loss_l, pred_labels_l = model((labeled_data, labeled))
        loss += loss_function(recon_batch_l, labeled_data, latent_loss_l, pred_labels_l, labeled)

        optimizer.zero_grad()

        loss.backward()
        # clip_grad_norm_(model.parameters(), 1.0)

        train_loss += loss.item()

        optimizer.step()
        scheduler.step()

        if batch_idx % args.log_interval == 0:

            labels = torch.unsqueeze(labels, 1)
            one_hot = torch.FloatTensor(labels.size()[0], NUM_CATEGORIES).zero_()
            one_hot.scatter_(1, labels, 1)

            pred_loss = -(one_hot*(pred_labels)).sum(dim=1).sum()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLab Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                pred_loss))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    print(f"\t\t {unlab}/{unlab+lab}")

def test(epoch):
    model.eval()
    test_loss = 0

    pred_loss = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            recon_batch, latent_loss, pred_labels = model((data, None))
            test_loss += loss_function(recon_batch, data, latent_loss, pred_labels, labels).item()

            labels = torch.unsqueeze(labels, 1)
            one_hot = torch.FloatTensor(labels.size()[0], NUM_CATEGORIES).zero_()
            one_hot.scatter_(1, labels, 1)

            pred_loss += -(one_hot * (pred_labels)).sum(dim=1).sum()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    pred_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('=========> Pred set loss: {:.4f}'.format(pred_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():

            sample = torch.randn(64, HIDDEN_DIM).to(device)
            labels = torch.zeros(64, NUM_CATEGORIES).to(device)
            labels[torch.arange(64), torch.arange(8).repeat(8)] = 1

            log_pis = -torch.log(torch.tensor(NUM_CATEGORIES).float().to(device))
            mu, logvar = torch.zeros_like(sample), torch.ones_like(sample)

            samp, log_det = model.get_latent_sample(sample, mu, logvar)
            img_sample = model.decode(samp, labels).cpu()

            save_image(img_sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
import pickle
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def plot_model(dset, model, constraint=lambda x: np.ones(len(x)).astype(bool), conditioning=False, index=0):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    data = dset.sample(5000)
    prior_samps = model.prior.sample([5000])

    if conditioning:
        data = data[data[:, index] == 1]

        zs, prior_logprob, log_det = model(data)
        xs, _ = model.sample_backwards(25000)

        categorical_variables = xs[:, :model.categorical_dims]
        continuous_variables = xs[:, model.categorical_dims:]
        x = continuous_variables[categorical_variables[:, index] == 1]

    else:
        zs, prior_logprob, log_det = model(data)
        xs, _ = model.backward(prior_samps)
        x = xs[-1]

    z = zs[-1]
    z = z.detach().numpy()

    x = x.detach().numpy()
    mask = constraint(x)
    print(f"# generated: {len(mask)}")
    print(f"# broken constraint: {(~mask).sum()}")
    print(f"% broken constraint: {(~mask).sum()/len(mask)}")

    axes[0].scatter(prior_samps[:, 0], prior_samps[:, 1], c='g', s=5, alpha=.5)
    axes[0].set_title('Prior samples')

    axes[1].scatter(z[:, 0], z[:, 1], c='g', s=5, alpha=.5)
    axes[1].set_title('x -> Z')

    axes[2].scatter(data[:, -2], data[:, -1], c='g', s=5, alpha=.5)
    axes[2].set_title('Data Samples')

    axes[3].scatter(x[:, 0][mask], x[:, 1][mask], c='g', s=5, alpha=.5, label='valid')
    axes[3].scatter(x[:, 0][~mask], x[:, 1][~mask], c='r', s=5, alpha=.5, label='invalid')
    axes[3].set_title('Z -> x')
    axes[3].legend(loc='best')

    for ax in axes:
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])

    plt.show()


def soft_lt(a, t): # a > t
    return -F.softplus(t-a, beta=1)

def soft_gt(a, t): # a > t
    return -F.softplus(a-t, beta=1)

def where(cond, x_1):
    return (cond.float() * x_1)

class generator:
    def plot(self, samples=512, categorical=False):
        plt.figure(figsize=(4, 4))
        x = self.sample(samples)
        if categorical:
            x, y = x[0], x[1]
            for y_ in np.unique(y.numpy()):
                plt.scatter(x[y==y_, 0], x[y==y_, 1], s=5, alpha=0.1)

        else:
            plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.1)

        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        # plt.axvline(0)
        # plt.axhline(0)
        plt.grid(True)
        plt.show()


class DatasetSIGGRAPH:
    """ 
    haha, found from Eric https://blog.evjang.com/2018/01/nf2.html
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """

    def __init__(self):
        with open('siggraph.pkl', 'rb') as f:
            xY = np.array(pickle.load(f), dtype=np.float32)
            xY -= np.mean(xY, axis=0)  # center
        self.xY = torch.from_numpy(xY)

    def sample(self, n):
        x = self.xY[np.random.randint(self.xY.shape[0], size=n)]
        return x


class DatasetMoons(generator):
    """ two half-moons """

    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)


class DatasetMixture(generator):
    """ 4 mixture of gaussians """

    def sample(self, n):
        assert n % 4 == 0
        r = np.r_[np.random.randn(n // 4, 2) * 0.5 + np.array([0, -2]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([0, 0]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([2, 2]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([-2, 2])]
        return torch.from_numpy(r.astype(np.float32))


class GaussianMixture(generator):
    """ 4 mixture of gaussians """

    def sample(self, n):
        self.NUM_CATEGORIES = 2

        assert n % 2 == 0
        r = np.r_[np.random.randn(n // 2, 2) * 1 + np.array([-1.5, 0]),
                  np.random.randn(n // 2, 2) * 1 + np.array([1.5, 0])]
        labels = np.concatenate((np.zeros(n//2), np.ones(n//2)), axis=0)

        labels = torch.unsqueeze(torch.from_numpy(labels.astype(int)), 1)
        one_hot = torch.FloatTensor(labels.size()[0], self.NUM_CATEGORIES).zero_()
        one_hot.scatter_(1, labels, 1)

        return torch.cat((one_hot, torch.from_numpy(r.astype(np.float32))), dim=-1)

    def plot(self, nsamples=5000):

        plt.figure(figsize=(4, 4))

        x_samps = self.sample(nsamples)

        for k in range(self.NUM_CATEGORIES):
            x = x_samps[x_samps[:, k] == 1]
            plt.scatter(x[:, -2], x[:, -1], s=5, alpha=0.5)

        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        plt.show()

class ConstrainedGaussian(generator):
    """ Gaussian with no center """

    def sample(self, n):
        r, fact = [], 2
        while len(r) < n:
            fact = fact ** 2
            r = np.random.randn(n * fact, 2) * 2
            r = r[r[:, 0] ** 2 + r[:, 1] ** 2 > 4]
        r = r[:n]
        return torch.from_numpy(r.astype(np.float32))


    def hard_constraint(self, x):
        constraint_broken = (x[:, 0] ** 2 + x[:, 1] ** 2 >= 4)
        return constraint_broken

    def constraint(self, x):
        magnitude = (4 - (x[:, 0] ** 2 + x[:, 1] ** 2))
        return magnitude * (~self.hard_constraint(x)).float()

    def eval_constraints(self, arr, data_indexes):
        broken_constraints = self.constraint(arr).view(-1, 1)
        return -(broken_constraints**2).view(-1,)


class ConstrainedGaussianInner(generator):
    """ Gaussian with no center """

    def rest0_x0(self, x, val=.5):
        return ((x[:, 0] > - val) & (x[:, 0] < val))

    def rest0_x1(self, x, val=4):
        return (x[:, 1] > - val) & (x[:, 1] < val)

    def rest0(self, x):
        return self.rest0_x0(x) & self.rest0_x1(x)

    def rest1_x0(self, x, val=.5):
        return ((x[:, 0] - x[:, 1] - val <= 0) &
                (x[:, 0] - x[:, 1] + val >= 0))

    def rest1_x1(self, x, val=4., val2=1.):
        return ((x[:, 0] + x[:, 1] - val <= 0) &
                (x[:, 0] + x[:, 1] + val >= 0))

    def rest1(self, x):
        return self.rest1_x0(x) & self.rest1_x1(x)

    def sample(self, n):
        r, fact = [], 2
        while len(r) < n:
            fact = fact ** 2
            r = np.random.randn(n * fact, 2) * 2
            r = r[self.true_constraint(torch.tensor(r).float()).detach().numpy()]
        r = r[:n]
        return torch.from_numpy(r.astype(np.float32))

    def true_constraint(self, x):
        constraint_valid = self.rest1_x0(x) & self.rest1_x1(x)
        return constraint_valid

    def hard_constraint(self, x):
        constraint_valid = self.rest1(x)
        return constraint_valid

    def constraint(self, x):
        # find the closest boundary:
        c1 = 0
        c2 = 2.
        c3 = 3.0
        c4 = 1.0

        magnitude = (
        torch.where((x[:, 0] - x[:, 1] > .5) & self.rest1_x1(x, val=4., val2=0.),
                    (x[:, 0] - x[:, 1] - c1),
                    torch.zeros(len(x))).view(-1,) +
        torch.where((x[:, 0] - x[:, 1] < -.5) & self.rest1_x1(x, val=4., val2=0.),
                    (- x[:, 0] + x[:, 1] - c1),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 0] + x[:, 1] > 4) & self.rest1_x0(x),
                    (x[:, 0] + x[:, 1] - c3),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 0] + x[:, 1] < -4) & self.rest1_x0(x),
                    (- x[:, 0] - x[:, 1] - c3),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 0] >= 2.25) & (x[:, 0] - x[:, 1] >= .5) & (x[:, 0]  + x[:, 1] >= 4),
                    (1.414*(x[:, 0] - c4)),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 0] <=-2.25) & (x[:, 0] - x[:, 1] <= -.5) & (x[:, 0] + x[:, 1] <=-4),
                    (1.414*(-x[:, 0] - c4)),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 1] >= 2.25) & (x[:, 0] - x[:, 1] <= -.5) & (x[:, 0] + x[:, 1] >= 4),
                    (1.414*(x[:, 1] - c4)),
                    torch.zeros(len(x))).view(-1, ) +
        torch.where((x[:, 1] <=-2.25) & (x[:, 0] - x[:, 1] >=  .5) & (x[:, 0] + x[:, 1] <=-4),
                    (1.414*(-x[:, 1] - c4)),
                    torch.zeros(len(x))).view(-1, )# +
        # torch.where((x[:, 0] + x[:, 1] <= 1) & (x[:, 0] + x[:, 1] > 0) & self.rest1_x0(x),
        #             (((-x[:, 0] - x[:, 1] + c2))),
        #             torch.zeros(len(x))).view(-1, ) +
        # torch.where((x[:, 0] + x[:, 1] >= -1) & (x[:, 0] + x[:, 1] <= 0) & self.rest1_x0(x),
        #             (((x[:, 0] + x[:, 1] + c2))),
        #             torch.zeros(len(x))).view(-1, ) +
        # torch.where((x[:, 0] - x[:, 1] > .5) & self.rest1_x1(x, val=1, val2=0),
        #             (x[:, 0] - x[:, 1] + c1 + 1),
        #             torch.zeros(len(x))).view(-1, ) +
        # torch.where((x[:, 0] - x[:, 1] < -.5) & self.rest1_x1(x, val=1, val2=0),
        #             (-x[:, 0] + x[:, 1] + c1 + 1),
        #             torch.zeros(len(x))).view(-1, )
        )
        return magnitude

    def eval_constraints(self, arr, data_indexes):
        broken_constraints = self.constraint(arr).view(-1, 1)
        return -(10*broken_constraints).view(-1,)**2


class WeirdConstrainedGaussian(generator):
    """ """

    def rest0(self, x):
        return (x[:, 1] > -2) & (x[:, 1] < 2) & (x[:, 0] > -2) & (x[:, 0] < 2)

    def rest1(self, x):
        return ~((x[:, 1] > -2) & (x[:, 1] < 2) & (x[:, 0] > -2) & (x[:, 0] < 2))

    def rest2(self, x):
        return ((x[:, 0] - x[:, 1] + 2 >= 0) & (x[:, 0] + x[:, 1] - 2 <= 0) &
                (-x[:, 0] + x[:, 1] + 2 >= 0) & (-x[:, 0] - x[:, 1] - 2 <= 0))

    def rest3(self, x):
        return ~((x[:, 0] - x[:, 1] + 2 >= 0) & (x[:, 0] + x[:, 1] - 2 <= 0) &
                 (-x[:, 0] + x[:, 1] + 2 >= 0) & (-x[:, 0] - x[:, 1] - 2 <= 0))

    def mag0(self, x):
        return (where((x[:, 1] <= -2), soft_lt(x[:, 1], -2)) +
                where((x[:, 1] > 2), soft_gt(x[:, 1], 2)) +
                where((x[:, 0] <= -2), soft_lt(x[:, 0], -2)) +
                where((x[:, 0] > 2), soft_gt(x[:, 0], 2))) * (~self.rest0(x)).float()

    def mag1(self, x):
        return (where(self.rest0(x) & (x[:, 0] - x[:, 1] > 0), soft_lt(x[:, 0] - x[:, 1], 2)) +
                where(self.rest0(x) & (x[:, 0] + x[:, 1] <= 0), soft_gt(x[:, 0] + x[:, 1], -2)) +
                where(self.rest0(x) & (-x[:, 0] + x[:, 1] >= 0), soft_lt(-x[:, 0] + x[:, 1], 2)) +
                where(self.rest0(x) & (-x[:, 0] - x[:, 1] < 0), soft_gt(-x[:, 0] - x[:, 1], -2))) * (
               ~self.rest1(x)).float()

    def mag2(self, x):
        return (where((x[:, 0] - x[:, 1] + 2 < 0), soft_lt(x[:, 0] - x[:, 1] + 2, 0)) +
                where((x[:, 0] + x[:, 1] - 2 > 0), soft_gt(x[:, 0] + x[:, 1] - 2, 0)) +
                where(-x[:, 0] + x[:, 1] + 2 < 0, soft_lt(-x[:, 0] + x[:, 1] + 2, 0)) +
                where(-x[:, 0] - x[:, 1] - 2 > 0, soft_gt(-x[:, 0] - x[:, 1] - 2, 0))) * (~self.rest2(x)).float()

    def mag3(self, x):
        return (where(self.rest2(x) & (x[:, 1] >= 0), soft_lt(x[:, 1], 2)) +
                where(self.rest2(x) & (x[:, 1] <= 0), soft_gt(x[:, 1], -2)) +
                where(self.rest2(x) & (x[:, 0] >= 0), soft_lt(x[:, 0], 2)) +
                where(self.rest2(x) & (x[:, 0] <= 0), soft_gt(x[:, 0], -2))) * (~self.rest3(x)).float()

    def eval_constraints(self, arr, data_indexes):
        broken_constraints = torch.cat(
            [self.mag0(arr).view(-1, 1), self.mag1(arr).view(-1, 1), self.mag2(arr).view(-1, 1),
             self.mag3(arr).view(-1, 1)], -1)
        return 5*torch.sum(broken_constraints * data_indexes, dim=-1)

    def hard_constraint(self, arr, data_indexes):
        return self.eval_constraints(arr, data_indexes) == 0

    def constraints(self, k, arr):
        return [self.rest0, self.rest1, self.rest2, self.rest3][k](arr)

    def sample(self, n):

        options = np.arange(4)
        r, fact = [], 2

        while len(r) < n:
            fact = fact ** 2

            r = np.random.randn(n * fact, 2) * 2
            k = np.random.choice(options, size=n * fact)

            rs = []
            min_len = (n // 4) + 1

            for k_ in range(4):
                r_ = r[k == k_]
                r_ = r_[self.constraints(k_, r_)]

                index = np.zeros((len(r_), 4))
                index[:, k_] = 1

                r_ = np.concatenate([index, r_], axis=-1)
                min_len = np.min([min_len, len(r_)])

                rs.append(r_)

            rs = [r[:min_len, :] for r in rs]
            r = np.concatenate(rs, axis=0)

        r = r[np.random.choice(len(r), size=len(r), replace=False)]
        return torch.from_numpy(r.astype(np.float32))

    def plot(self, nsamples=5000):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        x_samps = self.sample(nsamples)

        for k, ax in enumerate(axes):
            x = x_samps[x_samps[:, k] == 1]
            ax.scatter(x[:, -2], x[:, -1], s=5, alpha=0.5)
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])

        plt.show()

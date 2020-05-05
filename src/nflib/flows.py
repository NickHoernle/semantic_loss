"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal

from nflib.nets import LeafParam, MLP, ARMLP

class BaseDistribution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.zeros = torch.zeros(dim)
        self.ones = torch.eye(dim)
        self.base_dist = MultivariateNormal(self.zeros, self.ones)

    def log_prob(self, x):
        return self.base_dist.log_prob(x)

    def forward(self, x):
        return self.log_prob(x)

    def sample(self, n_samps):
        return self.base_dist.sample((n_samps,))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.zeros = self.zeros.to(*args, **kwargs)
        self.ones = self.ones.to(*args, **kwargs)
        self.base_dist = MultivariateNormal(self.zeros, self.ones)
        return self


class BaseDistributionMixtureGaussians(nn.Module):
    def __init__(self, dim, num_categories):
        super().__init__()
        self.zeros = torch.zeros(dim)
        self.ones = torch.eye(dim)

        base = MultivariateNormal(self.zeros, self.ones)
        self.means_ = nn.Parameter(torch.cat([base.sample((1,)) for s in range(num_categories)]))
        self.base_dist = MultivariateNormal(torch.zeros_like(self.means[0]), self.ones)
        self.num_categories = num_categories
        self.precision_ = 1

    @property
    def means(self):
        return self.means_

    @means.setter
    def means(self, means_):
        # weighted updating
        self.means_.data.copy_(0.9*self.means_.data + 0.1*means_.data)

    @property
    def precision(self):
        return self.precision_

    def anneal_precision(self):
        self.precision_ = np.min([self.precision_*1.2, self.num_categories])
        self.base_dist = MultivariateNormal(torch.zeros_like(self.means[0]), self.ones / self.precision_)

    def set_means_from_latent_data(self, latent_data, log_prob):
        log_prob = log_prob - torch.logsumexp(log_prob, dim=1).unsqueeze(dim=1)
        new_means = (latent_data.repeat(10, 1, 1).transpose(1, 0) * torch.exp(log_prob.unsqueeze(dim=-1))).mean(dim=0)
        self.means = new_means

    def set_means_from_latent_data_known_labels(self, latent_data, labels):

        labels = torch.unsqueeze(labels, 1)
        one_hot = torch.FloatTensor(latent_data.size()[0], self.num_categories).zero_()
        one_hot.scatter_(1, labels, 1)
        new_means = (latent_data.repeat(10, 1, 1).transpose(1, 0) * one_hot.unsqueeze(dim=-1)).mean(dim=0)
        self.means = new_means

    def log_prob(self, x, labels=[]):
        return self.base_dist.log_prob(x.unsqueeze(dim=1).repeat(1,self.num_categories,1) - self.means.repeat(len(x), 1, 1))

    def forward(self, x):
        return self.log_prob(x)

    def sample(self, n_samps, labels):
        return (labels.unsqueeze(dim=2)*(self.means.repeat(n_samps, 1, 1)
                    + self.base_dist.sample((n_samps,)).unsqueeze(dim=1).repeat(1,self.num_categories,1))).sum(dim=1)

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     self.zeros = self.zeros.to(*args, **kwargs)
    #     self.ones = self.ones.to(*args, **kwargs)
    #     self.base_dist = MultivariateNormal(self.zeros, self.ones)
    #     return self


class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True, **kwargs):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x, **kwargs):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, **kwargs):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x, **kwargs):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.parity = parity

        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)

        self.condition = kwargs.get('conditioning', False)
        self.num_condition = kwargs.get('num_conditioning', 0)

        if scale:
            self.s_cond = net_class(self.dim // 2 + self.num_condition*self.condition,
                                    self.dim // 2, nh, 3)
        if shift:
            self.t_cond = net_class(self.dim // 2 + self.num_condition*self.condition,
                                    self.dim // 2, nh, 3)

        self.cutter = nn.Hardtanh(-1,1)
        
    def forward(self, x, **kwargs):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0

        z0 = x0  # untouched half
        if self.condition:
            x0 = torch.cat((x0, kwargs['condition_variable']), -1)

        s = self.cutter(self.s_cond(x0))
        t = self.t_cond(x0)
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other

        if self.parity:
            z0, z1 = z1, z0

        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)

        return z, log_det
    
    def backward(self, z, **kwargs):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0

        x0 = z0  # this was the same

        if self.condition:
            z0 = torch.cat((z0, kwargs['condition_variable']), -1)

        s = self.cutter(self.s_cond(z0))
        t = self.t_cond(z0)

        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)

        return x, log_det

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     for i in range(self.dim):
    #         self.layers[str(i)] = self.layers[str(i)].to(*args, **kwargs)
    #     return self


class SlowMAF(nn.Module):
    """ 
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, **kwargs):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)

        self.condition = kwargs.get('conditioning', False)
        self.num_condition = kwargs.get('num_conditioning', 0)

        for i in range(1, dim):
            self.layers[str(i)] = net_class(i+self.num_condition*self.condition, 2, nh)

        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x, **kwargs):

        z = torch.zeros_like(x)
        log_det = torch.zeros_like(x[:,0])

        for i in range(self.dim):
            if self.condition:
                st = self.layers[str(i)](torch.cat((x[:, :i], kwargs['condition_variable']), -1))
            else:
                st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s

        return z, log_det

    def backward(self, z, **kwargs):

        x = torch.zeros_like(z)
        log_det = torch.zeros_like(z[:,0])

        for i in range(self.dim):
            if self.condition:
                st = self.layers[str(i)](torch.cat((x[:, :i], kwargs['condition_variable']), -1))
            else:
                st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s

        return x, log_det

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for i in range(self.dim):
            self.layers[str(i)] = self.layers[str(i)].to(*args, **kwargs)
        return self

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=MLP, nh=24, **kwargs):
        super().__init__()
        self.dim = dim
        self.parity = parity

        self.condition = kwargs.get('conditioning', False)
        self.num_condition = kwargs.get('num_conditioning', 0)

        self.net = net_class(dim + self.num_condition*self.condition, dim * 2, nh)

    def forward(self, x, **kwargs):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        x0 = x
        if self.condition:
            x0 = torch.cat((x, kwargs['condition_variable']), -1)
        st = self.net(x0)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, **kwargs):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros_like(z[:,0])
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            x0 = x.clone()
            if self.condition:
                x0 = torch.cat((x, kwargs['condition_variable']), -1)
            st = self.net(x0) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        original_dict = super().state_dict(destination, prefix, keep_vars)
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     for i in range(self.dim):
    #         self.layers[str(i)] = self.layers[str(i)].to(*args, **kwargs)
    #     return self

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())

        self.P = nn.Parameter(P, requires_grad=False) # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

        self.static_ones = torch.ones(self.dim)

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(self.static_ones)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, **kwargs):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z, **kwargs):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     original_dict = super().state_dict(destination, prefix, keep_vars)
    #     original_dict[prefix + 'P'] = self.P
    #     original_dict[prefix + 'L'] = self.L
    #     original_dict[prefix + 'S'] = self.S
    #     original_dict[prefix + 'U'] = self.U
    #     return original_dict
    #
    # def load_state_dict(self, state_dict, strict=True):
    #     """ Overrides state_dict() to load also theta value"""
    #
    #     print(state_dict.keys())
    #     print(state_dict)
    #
    #     P = state_dict.pop('P')
    #     L = state_dict.pop('L')
    #     S = state_dict.pop('S')
    #     U = state_dict.pop('U')
    #
    #     with torch.no_grad():
    #         self.P.copy_(P)
    #         self.L.copy_(L)
    #         self.S.copy_(S)
    #         self.U.copy_(U)
        # # import pdb
        # # pdb.set_trace()
        # # self.L.load_state_dict(state_dict)
        # # self.S.load_state_dict(state_dict)
        # # self.U.load_state_dict(state_dict)
        # # self.L.load_state_dict(state_dict)
        #
        # # self.register_buffer('P', P)
        # # self.register_buffer('L', torch.tensor(L))
        # # self.register_buffer('S', torch.tensor([S]))
        # # self.register_buffer('U', torch.tensor([U]))
        #
        # super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.P = self.P.to(*args, **kwargs)
        self.static_ones = self.static_ones.to(*args, **kwargs)
        return self

# class MixedOutputLayer(nn.Module):
#     def __init__(self, dim, cont_dim, categorical_dim):
#         super().__init__()
#         assert cont_dim + categorical_dim == dim
#
#         self.dim = dim
#         self.cont_dim = cont_dim
#         self.categorical_dim = categorical_dim
#
#     def forward(self, x):
#         z =
#         log_det = torch.sum(torch.log(torch.abs(self.S)))
#         return z, log_det
#
#     def backward(self, z):
#         W = self._assemble_W()
#         W_inv = torch.inverse(W)
#         x = z @ W_inv
#         log_det = -torch.sum(torch.log(torch.abs(self.S)))
#         return x, log_det

# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, **kwargs):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def __flow(self, func, input, **kwargs):
        '''
        func is either the forward or the backward call.
        input corresponds to x or z given the direction of the call.
        '''
        assert (func == 'fwd') or (func == 'bkwd')

        m, _ = input.shape
        log_det = torch.zeros_like(input[:,0])
        intermediate = [input]

        for k, flow in enumerate(self.flows[::(-1)**(func == 'bkwd')]):
            if func == 'fwd':
                input, ld = flow.forward(input, **kwargs)
            else:
                input, ld = flow.backward(input, **kwargs)
            log_det += ld
            intermediate.append(input)

        return intermediate, log_det

    def forward(self, x, **kwargs):
        return self.__flow('fwd', x, **kwargs)

    def backward(self, z, **kwargs):
        return self.__flow('bkwd', z, **kwargs)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     original_dict = super().state_dict(destination, prefix, keep_vars)
    #     return original_dict

    # def load_state_dict(self, state_dict, strict=True):
    #     from collections import OrderedDict
    #     for i, flow in enumerate(self.flows):
    #         these_keys = [str(k) for k in state_dict.keys() if f"flows.{i}." in k]
    #         dict = OrderedDict()
    #         for k in these_keys:
    #             dict[k.replace(f"flows.{i}.", "")] = state_dict[k]
    #
    #         print(dict)
    #         flow.load_state_dict(state_dict=dict, strict=True)
    #     # super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for f in self.flows:
            f.to(*args, **kwargs)
        return self


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows, **kwargs):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows, **kwargs)
    
    def forward(self, x, **kwargs):
        zs, log_det = self.flow.forward(x, **kwargs)
        prior_logprob = self.prior.forward(zs[-1])
        return zs, prior_logprob, log_det

    def backward(self, z, **kwargs):
        xs, log_det = self.flow.backward(z, **kwargs)
        return xs, log_det
    
    def sample(self, num_samples, **kwargs):
        z = self.prior.sample(num_samples)
        xs, lsdj = self.flow.backward(z, **kwargs)
        return xs[-1], lsdj

    def get_state_params(self):
        import copy
        return copy.deepcopy(self.flow.state_dict())

    def load_from_state_dict(self, state_dict):
        return self.flow.load_state_dict(state_dict)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.prior = self.prior.to(*args, **kwargs)
        self.flow = self.flow.to(*args, **kwargs)
        return self


class NormalizingFlowModelWithCategorical(NormalizingFlowModel):
    def __init__(self, prior, flows, categorical_dims, **kwargs):
        super().__init__(prior, flows, **kwargs)
        self.categorical_dims = categorical_dims
        self.logits = nn.Parameter(torch.zeros((1,categorical_dims)).normal_())
        self.loss_fn = nn.NLLLoss()

    def forward(self, x, **kwargs):
        tau = kwargs.get('tau', 1)
        categorical_variables = x[:,:self.categorical_dims]
        continuous_variables = x[:,self.categorical_dims:]

        zs, log_det = self.flow.forward(continuous_variables, condition_variable=categorical_variables)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)

        logits = self.logits.repeat(x.size()[0], 1)
        categorical_samples = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

        categorical_nll = -(categorical_samples*(logits)).sum(dim=1)

        return zs, prior_logprob + categorical_nll, log_det

    def backward(self, z, **kwargs):
        # first sample the categrical variable then condition on it
        categorical_variables = z[:, :self.categorical_dims]
        continuous_variables = z[:, self.categorical_dims:]

        xs, log_det = self.flow.backward(continuous_variables, condition_variable=categorical_variables)
        return xs, log_det

    def sample(self, num_samples, **kwargs):
        xs, _ = self.sample_backwards(num_samples, **kwargs)
        return xs

    def sample_backwards(self, num_samples, **kwargs):
        tau = kwargs.get('tau', 1)
        logits = self.logits.repeat(num_samples, 1)
        categorical_samples = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1).detach()

        # sample from the prior
        z = self.prior.sample((num_samples,))
        xs, log_det = self.flow.backward(z, condition_variable=categorical_samples)
        return torch.cat((categorical_samples, xs[-1]), dim=-1), log_det

    def forward_categorical(self, x, **kwargs):
        logits = self.logits.repeat(x.size()[0], 1)
        categorical_samples = F.gumbel_softmax(logits, tau=.5, hard=False, dim=-1)
        return categorical_samples


class SS_Flow(nn.Module):
    def __init__(self, flows, NUM_CATEGORIES, dims):
        super().__init__()
        self.flow_main = flows
        self.tau = 2
        self.NUM_CATEGORIES = NUM_CATEGORIES
        self.num_dims = dims

    @property
    def prior(self):
        return self.flow_main.prior

    @classmethod
    def dequantize(cls, x):
        x = (x * 255. + torch.rand_like(x)) / 256.
        return x

    @classmethod
    def to_logits(cls, x):
        """Convert the input image `x` to logits.

        Args:
            x (torch.Tensor): Input image.
            sldj (torch.Tensor): Sum log-determinant of Jacobian.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        bounds = torch.tensor([0.9], dtype=torch.float32)
        y = (2 * x - 1) * bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - bounds).log() - bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj

    def forward(self, sample, **kwargs):

        (data, labels) = sample
        # data = SS_Flow.dequantize(data)

        if type(labels) != type(None):
            return self.forward_labelled(x=data.view(-1, self.num_dims), y=labels, **kwargs)

        return self.forward_unlabelled(x=data.view(-1, self.num_dims), **kwargs)

    def forward_unlabelled(self, x, **kwargs):

        # x, sldj = SS_Flow.to_logits(x)
        zs, prior_logprob, log_det = self.flow_main(x, **kwargs)

        # sample label
        prior = torch.log(torch.tensor(1.0 / self.NUM_CATEGORIES))

        # gumbel-softmax
        sampled_labels = F.gumbel_softmax(prior_logprob, tau=self.tau, hard=False)

        # prediction loss
        log_pred_label_sm = torch.log(torch.softmax(prior_logprob, dim=1) + 1e-10)
        latent = log_det.sum()#-sldj.sum()

        return zs[-1],\
            (-(prior_logprob*sampled_labels).sum(dim=1).sum(), -(latent), log_pred_label_sm, prior), \
               sampled_labels

    def forward_labelled(self, x, y, **kwargs):

        one_hot = y
        x, sldj = SS_Flow.to_logits(x)

        zs, prior_logprob, log_det = self.flow_main(x, **kwargs)

        # prediction
        log_pred_label_sm = torch.log(torch.softmax(prior_logprob, dim=1) + 1e-10)

        return zs[-1],\
               (-(prior_logprob*one_hot).sum(dim=1).sum(), -(log_det.sum()-sldj.sum()), log_pred_label_sm), \
               one_hot

    def backward(self, labels, **kwargs):
        z = self.flow_main.prior.sample(len(labels), labels)
        xs, log_det_back = self.flow_main.backward(z, **kwargs)
        return xs[-1], log_det_back

    def get_state_params(self):
        import copy
        import collections
        flow_params = copy.deepcopy(self.flow_main.get_state_params())
        prior_params = copy.deepcopy(self.flow_main.prior.state_dict())
        return collections.OrderedDict([('flow_params', flow_params), ('prior_params', prior_params)])

    def load_from_state_dict(self, state_dict):
        flow_params = state_dict.pop('flow_params')
        self.flow_main.load_from_state_dict(flow_params)
        prior_params = state_dict.pop('prior_params')
        self.flow_main.prior.load_state_dict(prior_params)
        return True

    def sample_labelled(self, labels):
        return self.backward(labels)

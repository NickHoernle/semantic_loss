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

from nflib.nets import LeafParam, MLP, ARMLP

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
        
    def forward(self, x, **kwargs):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0

        z0 = x0  # untouched half
        if self.condition:
            x0 = torch.cat((x0, kwargs['condition_variable']), -1)

        s = self.s_cond(x0)
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

        s = self.s_cond(z0)
        t = self.t_cond(z0)

        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)

        return x, log_det


class SlowMAF(nn.Module):
    """ 
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

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
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
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
        log_det = torch.zeros(m)
        intermediate = [input]

        for flow in self.flows[::(-1)**(func == 'bkwd')]:
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

class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows, **kwargs):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows, **kwargs)
    
    def forward(self, x, **kwargs):
        zs, log_det = self.flow.forward(x, **kwargs)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, **kwargs):
        xs, log_det = self.flow.backward(z, **kwargs)
        return xs, log_det
    
    def sample(self, num_samples, **kwargs):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z, **kwargs)
        return xs

    def get_state_params(self):
        import copy
        return copy.deepcopy(self.flow.state_dict())

    def load_from_state_dict(self, state_dict):
        return self.flow.load_state_dict(state_dict)


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
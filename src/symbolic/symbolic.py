import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class ConstantEqualityGenerative(nn.Module):
    def __init__(
        self,
        ixs_active,
        ixs_inactive,
        **kwargs
    ):
        super(ConstantEqualityGenerative, self).__init__()
        self.ixs_active = ixs_active
        self.ixs_inactive = ixs_inactive

    def forward(self, x):
        ll1, ll2, ll3, lp1, lp2, lp3 = x

        lr = 0
        for i, (ll, lp) in enumerate([[ll1, lp1], [ll2, lp2], [ll3, lp3]]):
            sll1 = ll[:, self.ixs_active[i]]
            lr += sll1

        return lr / 3  # - lp3.log_softmax(dim=1)[:, self.ixs_active[2]]


class GEQConstant(nn.Module):
    def __init__(
            self,
            ixs1,
            ixs_not,
            ixs_less_than,
            threshold_upper,
            threshold_lower,
            threshold_limit,
            **kwargs
    ):
        super(GEQConstant, self).__init__()

        self.ixs1 = ixs1
        self.ixs_neg = ixs_less_than
        self.ixs_not = ixs_not

        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.threshold_limit = threshold_limit

        self.forward_transform = self.ixs1 + self.ixs_neg
        self.reverse_transform = np.argsort(self.forward_transform)

    def threshold1p(self):
        if self.threshold_lower > self.threshold_limit:
            self.threshold_lower -= 1

    def forward(self, x):

        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_neg]
        split3 = x[:, self.ixs_not]

        restricted1 = F.softplus(split1) + self.threshold_upper
        restricted2 = (split2 / split2).detach() * self.threshold_lower

        return torch.cat((restricted1, restricted2, split3), dim=1)[
            :, self.reverse_transform
        ]


class GEQ_Interaction(nn.Module):
    def __init__(
        self,
        ixs1,
        ixs_less_than,
        weights,
        intercept,
        threshold_lower=-10,
        device="cuda",
    ):
        super(GEQ_Interaction, self).__init__()
        self.ixs1 = ixs1
        self.ixs_less_than = ixs_less_than

        self.weights = weights
        self.intercept = intercept
        self.threshold_lower = threshold_lower

        self.forward_transform = self.ixs1 + self.ixs_less_than
        self.reverse_transform = np.argsort(self.forward_transform)

        self.device = device

    def threshold1p(self):
        if self.threshold_lower > -10:
            self.threshold_lower -= 1

    def forward(self, x):
        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_less_than]

        split1 = F.softplus(split1 - self.threshold_lower) + \
            self.threshold_lower
        # find the perpendicular vector to the line defined by weights
        a = -torch.tensor(self.weights).float().to(self.device).unsqueeze(1)
        dot_prod = split1.mm(a)
        distance = (dot_prod - self.intercept) / torch.norm(a)
        corrected_distance = -F.softplus(-distance)

        # find the point on the plane that is closest to the point specified
        pp_ = (dot_prod - self.intercept) / (torch.norm(a) ** 2)
        plane_point = split1 - pp_ * (a.transpose(0, 1))

        # distance along direction vector
        corrected_distance_vec = corrected_distance * (
            a.transpose(0, 1) / torch.norm(a)
        )
        new_point = plane_point + corrected_distance_vec

        restricted1 = new_point
        restricted2 = torch.ones_like(split2) * self.threshold_lower

        return torch.cat((restricted1, restricted2), dim=1)[:, self.reverse_transform]


class Between(nn.Module):
    def __init__(
        self,
        ixs1,
        ixs_less_than,
        threshold_upper=[-1.0, 1.0],
        threshold_lower=-10,
        **kwargs
    ):
        super(Between, self).__init__()
        self.ixs1 = ixs1
        self.ixs_less_than = ixs_less_than

        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower

        self.forward_transform = self.ixs1 + self.ixs_less_than
        self.reverse_transform = np.argsort(self.forward_transform)

    def threshold1p(self):
        pass

    def forward(self, x):

        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_less_than]

        greater_than = (
            F.softplus(
                split1 - self.threshold_upper[0]) + self.threshold_upper[0]
        )
        less_than = (
            -F.softplus(-greater_than + self.threshold_upper[1])
            + self.threshold_upper[1]
        )

        restricted2 = - \
            F.softplus(-split2 + self.threshold_lower) + self.threshold_lower

        return torch.cat((less_than, restricted2), dim=1)[:, self.reverse_transform]


class Box(nn.Module):
    def __init__(
        self,
        constrained_ixs,
        not_constrained_ixs,
        lims=((0.25, -0.25), (0.5, 5.5)),
        **kwargs
    ):
        super(Box, self).__init__()
        self.constrained_ixs = constrained_ixs
        self.not_constrained_ixs = not_constrained_ixs
        self.lims = torch.tensor(lims).float()

        assert len(lims) == len(constrained_ixs)

        self.forward_transform = self.constrained_ixs + self.not_constrained_ixs
        self.reverse_transform = np.argsort(self.forward_transform)

    def valid(self, x):
        split1 = x[:, self.constrained_ixs]
        return (
            (split1 + 1e-5 >= self.lims[:, 0][None, :])
            & (split1 - 1e-5 <= self.lims[:, 1][None, :])
        ).all(dim=-1)

    def forward(self, x, m=0.5):
        split1 = x[:, self.constrained_ixs]
        split2 = x[:, self.not_constrained_ixs]

        greater_than = F.softplus(split1 - self.lims[:, 0][None, :])
        offset = torch.log(
            torch.exp(self.lims[:, 1][None, :] - self.lims[:, 0][None, :]) - 1
        )
        less_than = -F.softplus(-greater_than + offset) + \
            self.lims[:, 1][None, :]

        return torch.cat((less_than, split2), dim=1)[:, self.reverse_transform]


class RotatedBox(Box):
    def __init__(
        self,
        constrained_ixs,
        not_constrained_ixs,
        lims=((0.25, -0.25), (0.5, 5.5)),
        theta=0.0,
        **kwargs
    ):
        super(RotatedBox, self).__init__(
            constrained_ixs, not_constrained_ixs, lims)
        self.theta = theta
        self.inverse_rotation_matrix = torch.tensor(
            [
                [np.cos(-theta), -np.sin(-theta)],
                [np.sin(-theta), np.cos(-theta)],
            ]
        ).float()
        self.rotation_matrix = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        ).float()

    def rotate(self, x):
        return x.mm(self.rotation_matrix)

    def rotate_inverse(self, x):
        return x.mm(self.inverse_rotation_matrix)

    def valid(self, x):
        split1 = x[:, self.constrained_ixs]
        return (
            (self.rotate(split1) + 1e-5 >= self.lims[:, 0][None, :])
            & (self.rotate(split1) - 1e-5 <= self.lims[:, 1][None, :])
        ).all(dim=-1)

    def forward(self, x, m=0.5):
        split1 = x[:, self.constrained_ixs]
        split2 = x[:, self.not_constrained_ixs]

        greater_than = F.softplus(
            self.rotate_inverse(split1) - self.lims[:, 0][None, :]
        )
        offset = torch.log(
            torch.exp(self.lims[:, 1][None, :] - self.lims[:, 0][None, :]) - 1
        )
        less_than = -F.softplus(-greater_than + offset) + \
            self.lims[:, 1][None, :]

        return torch.cat((self.rotate(less_than), split2), dim=1)[
            :, self.reverse_transform
        ]


class SinRelation(nn.Module):
    def __init__(self, constrained_ixs, constrained_to, not_constrained_ixs, **kwargs):
        super(SinRelation, self).__init__()
        self.constrained_ixs = constrained_ixs
        self.constrained_to = constrained_to
        self.not_constrained_ixs = not_constrained_ixs

        self.restriction = Box(
            constrained_ixs=constrained_ixs,
            not_constrained_ixs=constrained_to + not_constrained_ixs,
            lims=[(-0.1, 0.1)],
        )

        self.forward_transform = (
            self.constrained_ixs + self.constrained_to + self.not_constrained_ixs
        )
        self.reverse_transform = np.argsort(self.forward_transform)

    def forward(self, x):
        restricted = self.restriction(x)
        split1 = restricted[:, self.constrained_ixs]
        split2 = restricted[:, self.constrained_to]
        split3 = restricted[:, self.not_constrained_ixs]

        return torch.cat((split1 + torch.sin(split2), split2, split3), dim=1)[
            :, self.reverse_transform
        ]


class Identity(GEQConstant):
    def forward(self, x):
        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_neg]
        split3 = x[:, self.ixs_not]

        restricted2 = -F.softplus(-split2)

        return torch.cat((split1, restricted2, split3), dim=1)[
            :, self.reverse_transform
        ]


class OrList(nn.Module):
    def __init__(self, terms):
        super().__init__()
        self.layers = nn.ModuleList(terms)

    def threshold1p(self):
        for layer in self.layers:
            layer.threshold1p()

    def all_predictions(self, x):
        return torch.stack([f(x) for f in self.layers], dim=1)

    def forward(self, x, class_prediction, test=False):
        pred = self.all_predictions(x)
        log_py = class_prediction.log_softmax(dim=1)

        if test:
            return pred[np.arange(len(log_py)), log_py.argmax(dim=1)]

        return pred, log_py

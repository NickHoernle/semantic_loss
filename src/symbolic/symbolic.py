import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


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

        self.forward_transform = self.ixs1 + self.ixs_neg + self.ixs_not
        self.reverse_transform = np.argsort(self.forward_transform)

        self.fc = nn.Linear(len(self.forward_transform), len(self.forward_transform))

    def threshold1p(self):
        if self.threshold_lower > self.threshold_limit:
            self.threshold_lower -= 1

    def forward(self, x):
        x = self.fc(x)

        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_neg]
        split3 = x[:, self.ixs_not]

        restricted1 = F.softplus(split1 - self.threshold_upper) + self.threshold_upper
        # restricted2 = -F.softplus(-split2+self.threshold_lower)+self.threshold_lower
        restricted2 = torch.ones_like(split2) * self.threshold_lower

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

        split1 = F.softplus(split1 - self.threshold_lower) + self.threshold_lower
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

        # self.fc = nn.Linear(len(self.forward_transform), len(self.forward_transform))

    def threshold1p(self):
        pass

    def forward(self, x):

        split1 = x[:, self.ixs1]
        split2 = x[:, self.ixs_less_than]

        greater_than = (
            F.softplus(split1 - self.threshold_upper[0]) + self.threshold_upper[0]
        )
        less_than = (
            -F.softplus(-greater_than + self.threshold_upper[1])
            + self.threshold_upper[1]
        )

        restricted2 = -F.softplus(-split2 + self.threshold_lower) + self.threshold_lower

        return torch.cat((less_than, restricted2), dim=1)[:, self.reverse_transform]


#
#
# class GEQ(GEQConstant):
#
#     def forward(self, x):
#         x_ = x[:, self.forward_transform]
#         split1, split2, split3 = x_.split([len(self.ixs_pos), len(self.ixs_neg), len(self.ixs_not)], dim=1)
#
#         max_val = F.relu(split2).max(dim=1)[0]
#         min_val = (F.relu(-split1).max(dim=1)[0])
#
#         return torch.cat((split1 + min_val.unsqueeze(1) + max_val.unsqueeze(1),
#                           split2 - self.limit_threshold,
#                           split3), dim=1)[:, self.reverse_transform]
#
#
# class AndDisjoint(nn.Module):
#     def __init__(self, term1, term2):
#         super(AndDisjoint, self).__init__()
#         self.term1 = term1
#         self.term2 = term2
#
#     def forward(self, x):
#         return self.term2(self.term1(x))


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
        super(OrList, self).__init__()
        self.layers = nn.ModuleList(terms)
        self.fc = nn.Linear(2 * len(terms), len(terms))

    def threshold1p(self):
        for layer in self.layers:
            layer.threshold1p()

    def forward(self, x, class_prediction, test=False):
        pred = torch.stack([f(x) for f in self.layers], dim=1)
        log_py = self.fc(
            torch.cat((class_prediction, pred.max(dim=-1)[0]), dim=1)
        ).log_softmax(dim=1)
        # log_py = class_prediction.log_softmax(dim=1)

        if test:
            return pred[np.arange(len(log_py)), log_py.argmax(dim=1)]

        return pred, log_py

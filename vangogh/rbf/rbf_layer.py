import torch
import torch.nn as nn


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func=gaussian):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))

        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)

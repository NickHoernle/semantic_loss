import torch
from torch import nn

from .resnet import resnet, data_parallel

class RESNET_VAE(nn.Module):
    def __init__(self, depth=10, width=2, hidden_dim=10, num_output=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        f, params = resnet(depth, width, hidden_dim * 2)
        self.resnet_params = params
        self.resnet_function = f

        self.device_ids = kwargs.get("device_ids", 0)
        self.num_gpus = kwargs.get("device_ids", 1)

        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, num_output)
        )

    @property
    def encoder_params(self):
        return [v for v in self.resnet_params.values() if v.requires_grad]

    @property
    def decoder_params(self):
        return list(self.decoder.parameters())

    def to(self, device):
        self.decoder.to(device)

    def parameters(self):
        return self.encoder_params + self.decoder_params

    def encode(self, x):
        y = data_parallel(self.resnet_function,
                          x,
                          self.resnet_params,
                          True,
                          list(range(self.num_gpus))
                          ).float()
        return y[:, :self.hidden_dim], y[:, self.hidden_dim:]

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x = self.decode(z)
        return {
            "x": x,
            "z": z,
            "latent_params": (mu, logvar)
        }
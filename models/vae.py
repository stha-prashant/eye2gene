import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, device='cuda:0', code_size:int=2):
        self.code_size = code_size
        super(VariationalAutoEncoder, self).__init__()
        self.encoder =  nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(12288, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
        )
        self.mu_linear = nn.Linear(12, code_size)
        self.sigma_linear = nn.Linear(12, code_size)
        
        self.decoder =  nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 12288),
            nn.Sigmoid()
        )
        self.N = torch.distributions.Normal(0, torch.tensor(1.0))
        # if device == "cuda":
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)   

    def forward(self, x):
        x = self.encoder(x)
        latent_space_mean = self.mu_linear(x)
        latent_space_sigma = torch.exp(self.sigma_linear(x))
        z = latent_space_mean + latent_space_sigma * self.N.sample(latent_space_mean.shape)
        # TODO: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        self.kl = (latent_space_sigma ** 2 + latent_space_mean ** 2 - torch.log(latent_space_sigma) - 0.5).sum()
        op = self.decoder(z)
        return torch.reshape(op, (-1, 3, 64, 64))

    def get_latent_space(self, x):
        x = self.encoder(x)
        latent_space_mean = self.mu_linear(x)
        latent_space_sigma = torch.exp(self.sigma_linear(x))
        z = latent_space_mean + latent_space_sigma * self.N.sample(latent_space_mean.shape)
        return z
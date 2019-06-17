import torch
import torch.distributions as tdist
import math

class GaussianDistribution():
    def sample(mu):
        # for simplify
        std = torch.ones(mu.size(), dtype=torch.float32)
        return tdist.Normal(mu, std).sample()

    def log_density(act, mu):
        std = torch.ones(mu.size(), dtype=torch.float32)
        logstd = torch.log(std)
        return -(act-mu)*(act-mu) / (2*std*std) - \
                0.5*math.log(2*math.pi) - logstd

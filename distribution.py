import torch
import torch.distributions as tdist
import math

class DistributionInterface():
    def sample(self, mu, state):
        assert(False)
    
    def log_density(self, act, mu, state):
        assert(False)

    def train(self):
        pass

    def eval(self):
        pass

    def apply_loss(self, loss, retain_graph=False):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def set_ckpt(self, ckpt):
        pass

    def get_ckpt(self):
        return {}

class FixedGaussianDistribution(DistributionInterface):
    def sample(self, mu, state):
        # for simplify
        std = torch.ones(mu.size(), dtype=torch.float32)
        return tdist.Normal(mu, std).sample()

    def log_density(self, act, mu, state):
        std = torch.ones(mu.size(), dtype=torch.float32)
        logstd = torch.log(std)
        return -(act-mu)*(act-mu) / (2*std*std) - \
                0.5*math.log(2*math.pi) - logstd

class NetGaussianDistribution(DistributionInterface):
    def __init__(self, network, opt):
        self._network = network
        self._opt = opt

    def sample(self, mu, state):
        logstd = self._network(state)
        std = torch.exp(logstd)
        print(torch.mean(std))
        return tdist.Normal(mu, std).sample()
    
    def log_density(self, act, mu, state):
        logstd = self._network(state)
        std = torch.exp(logstd)
        return -(act-mu)*(act-mu) / (2*std*std) - \
                0.5*math.log(2*math.pi) - logstd

    def train(self):
        self._network.train()

    def eval(self):
        self._network.eval()

    def zero_grad(self):
        self._opt.zero_grad()

    def step(self):
        self._opt.step()

    def apply_loss(self, loss, retain_graph=False):
        self._opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._opt.step()

    def set_ckpt(self, ckpt):
        assert('netdist' in ckpt)
        self._network.load_state_dict(ckpt['netdist'])

    def get_ckpt(self):
        return {'netdist' : self._network.state_dict()}

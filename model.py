import torch

# Actor has:
# policy network(network)
# network optimizer(opt)
# policy distribution(distribution)

# Actor do:
# decide action at state from distribution (get_action)
# calculate density of policy from distribution (log_policy)
# apply loss from optimizer (apply_loss)

class Actor():
    def __init__(self, network, opt, distribution): 
        self._network = network
        self._opt = opt
        self._dist = distribution

    def get_action(self, state):
        mu = self._network(state)
        return self._dist.sample(mu)
        
    # return log_density of policy
    # input: states(torch.Tensor(state array: size = [:][state])
    # actions(torch.Tensor(action array: size = [:][action])
    # return: density(torch.Tensor(float array: size = [:][1]))
    def log_policy(self, states, actions):
        mu = self._network(states)
        res = self._dist.log_density(actions, mu)
        for _ in range(res.dim()-2): res._sum(1)
        return res.sum(1, keepdim=True)

    def mode_train(self):
        self._network.train()

    def mode_eval(self):
        self._network.eval()

    def apply_loss(self, loss, retain_graph=False):
        self._opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._opt.step()

# Critic has:
# policy network(network)
# network optimizer(opt)

# Critic do:
# decide value at state (get_value)
# apply loss from optimizer (apply_loss)

class Critic():
    def __init__(self, network, opt):
        self._network = network
        self._opt = opt

    def get_values(self, state):
        value = self._network(state)
        return value
        
    def mode_train(self):
        self._network.train()

    def mode_eval(self):
        self._network.eval()

    def apply_loss(self, loss, retain_graph=False):
        self._opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._opt.step()

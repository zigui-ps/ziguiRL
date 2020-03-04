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
		self.network = network
		self.opt = opt
		self.dist = distribution

	def get_action(self, state):
		mu = self.network(state.cuda()).cpu()
		return self.dist.sample(mu, state)
		
	def get_action_nodist(self, state):
		mu = self.network(state.cuda()).cpu().detach()
		return mu

	def get_std(self, state):
		return self.dist.get_scale(state)
		
	def evaluate(self, state, action, detach = False):
		mu = torch.squeeze(self.network(state.cuda()).cpu())
		action_logprobs = self.dist.log_prob(torch.squeeze(action), mu, state)
		dist_entropy = self.dist.entropy(mu, state)
		
		if detach: return action_logprobs.detach(), dist_entropy.detach()
		else: return action_logprobs, dist_entropy

	def train(self):
		self.network.train()
		self.dist.train()

	def eval(self):
		self.network.eval()
		self.dist.eval()

	def zero_grad(self):
		self.opt.zero_grad()
		self.dist.zero_grad()

	def step(self):
		self.opt.step()
		self.dist.step()

	def set_ckpt(self, ckpt):
		assert('actor' in ckpt)
		self.network.load_state_dict(ckpt['actor'])
		self.dist.set_ckpt(ckpt['dist'])

	def get_ckpt(self):
		ckpt = {'actor' : self.network.state_dict(),
				'dist' : self.dist.get_ckpt()}
		return ckpt

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
		value = self._network(state.cuda()).cpu()
		return value
		
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
		assert('critic' in ckpt)
		self._network.load_state_dict(ckpt['critic'])

	def get_ckpt(self):
		return {'critic' : self._network.state_dict()}

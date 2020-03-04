import torch
import torch.optim as optim
import torch.nn.functional as F

import policy
import deepnetwork
from model import Actor, Critic
import distribution
import customenv.customgym as customgym
import os, argparse
from statemodifier import *

model_path = os.path.join(os.getcwd(),'save_model')

def argument_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--train', type=int, default=0)
	parser.add_argument('--name', type=str, default='zigui')
	parser.add_argument('--env', type=str, default=None)
	parser.add_argument('--render', default=False, action="store_true")

	return parser.parse_args()

def PPO_agent_with_param(env, actor_network, actor_lr, critic_network, critic_lr, critic_decay, arg, render, zeroInit=False):
	o_size = env[0].observation_space.shape[0]
	a_size = env[0].action_space.shape[0]
	
	actor_network = deepnetwork.CNN([o_size] + actor_network + [a_size], "actor", zeroInit).cuda()
	actor_opt = optim.Adam(actor_network.parameters(), lr=actor_lr)
	
	dist = distribution.FixedGaussianDistribution(torch.distributions.normal.Normal, torch.ones(a_size))
	
	actor = Actor(actor_network, actor_opt, dist)
	
	critic_network = deepnetwork.CNN([o_size] + critic_network + [1], "critic").cuda()
	critic_opt = optim.Adam(critic_network.parameters(), lr=critic_lr, weight_decay=critic_decay)
	
	critic = Critic(critic_network, critic_opt)
	
	agent = policy.PPOAgent(env, actor, critic, arg, render)
	return agent

def SAC_agent_with_param(env, actor_network, actor_lr, critic_network, critic_lr, critic_decay, arg, render, zeroInit=False):
	o_size = env[0].observation_space.shape[0]
	a_size = env[0].action_space.shape[0]
	
	actor_network = deepnetwork.CNN([o_size] + actor_network + [a_size], "actor", zeroInit).cuda()
	actor_opt = optim.Adam(actor_network.parameters(), lr=actor_lr)
	
	dist = distribution.FixedGaussianDistribution(torch.distributions.normal.Normal, torch.ones(a_size))
	
	actor = Actor(actor_network, actor_opt, dist)
	
	critic_network = [deepnetwork.CNN([o_size] + critic_network + [1], "critic").cuda() for _ in range(2)]
	critic_opt = [optim.Adam(critic_network[i].parameters(), lr=critic_lr, weight_decay=critic_decay) for i in range(2)]
	critic = [Critic(critic_network[i], critic_opt[i]) for i in range(2)]
	
	agent = policy.SACAgent(env, actor, critic, arg, render)
	return agent

def train(agent, train_step, env_name, args_name, value_only = False):
	for i in range(train_step):
		agent.train(5, os.path.join(model_path, 'ckpt_{}_{}_max'.format(env_name, args_name)), value_only = value_only)
		torch.save(agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_{}'.format(env_name, args_name, i)))
		torch.save(agent.get_ckpt(), os.path.join(model_path, 'ckpt_{}_{}_latest'.format(env_name, args_name)))
		if agent.actor.dist.scale[0] >= 0.01: agent.actor.dist.scale *= 0.99
		print("dist : {}".format(agent.actor.dist.scale))

def demo(env, agent, render = 1):
	state = env.reset()
	t = 0
	while True:
		if t%render == 0: env.render()
		t += 1

		action = agent.next_action_nodist(state)
		state, reward, done, _, _ = env.step(action)
		if done:
			print("END")

def train_gym(args):
	env = [customgym.PythonGym(args.env, 1000)]
	agent = PPO_agent_with_param(env, [128, 128], 1e-4, [128, 128], 1e-4, 7e-4, \
		{'gamma':0.994, 'lamda':0.99, 'steps':2048, 'batch_size':32}, args.render)
	if args.load_model is not None: agent.set_ckpt(torch.load(os.path.join(os.getcwd(), args.load_model)))
	train(agent, args.train, env[0].name, args.name)
	if args.render: demo(env[0], agent)

if __name__ == "__main__":
	args = argument_parse()
	train_gym(args)
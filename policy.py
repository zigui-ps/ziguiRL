from statemodifier import ClassicModifier
from replaybuffer import ReplayBuffer
import torch
import os
import numpy as np

class AgentInterface():
	def __init__(self, env, actor, critic, args, render):
		self.env = env
		self.actor = actor
		self.critic = critic
		self._episodes = 0
		self.render = render
		self.state_modifier = ClassicModifier()
		self.zero_state = env[0].reset()
		self.maxscore = 0

		assert('steps' and 'batch_size' in args)
		self.steps = args['steps']
		self.batch_size = args['batch_size']

	def get_replay_buffer(self, gamma, env):
		total_score, steps, n = 0, 0, 0
		replay_buffer = ReplayBuffer()
		state = self.state_modifier.apply(env.reset())
		while steps < self.steps:
			self._episodes += 1
			n += 1
			if n == 1: print("0 state value {}".format(self.critic.get_values(state).detach()[0]))
			score = 0

			while True: # timelimits
				if self.render: env.render()
				action = self.actor.get_action(state).detach()
				next_state, reward, done, tl, _ = env.step(action)
				next_state = self.state_modifier.apply(next_state)
				
				if tl == 1: reward += self.critic.get_values(next_state).detach()[0] * gamma
				
				score += reward
				replay_buffer.append(state, action, reward, done == 1)
					
				state = next_state
				total_score, steps = total_score + reward, steps + 1
				if done == 1: break
		
		print("episodes: {}, score: {}, avg steps: {}, avg reward {}".format(self._episodes, total_score / n, steps / n, total_score / steps))
		return replay_buffer, total_score / n

	def train(self):
		assert(False)

	def next_action(self, state):
		state = self.state_modifier.modify(state)
		return self.actor.get_action(state.cuda()).cpu()

	def next_action_nodist(self, state):
		state = self.state_modifier.modify(state)
		return self.actor.get_action_nodist(state.cuda()).cpu()

	def get_ckpt(self):
		ckpt = {'episodes' : self._episodes, 'actor' : self.actor.get_ckpt(), \
			'critic' : self.critic.get_ckpt(), 'state_modifier' : self.state_modifier.get_ckpt(), 'maxscore' : self.maxscore}
		return ckpt

	def set_ckpt(self, ckpt):
		self._episodes = ckpt['episodes']
		#self.maxscore = ckpt['maxscore']
		self.actor.set_ckpt(ckpt['actor'])
		self.critic.set_ckpt(ckpt['critic'])
		self.state_modifier.set_ckpt(ckpt['state_modifier'])

class PPOAgent(AgentInterface):
	def __init__(self, env, actor, critic, args, render):
		super(PPOAgent, self).__init__(env, actor, critic, args, render)
		assert('gamma' and 'lamda' in args)
		self.gamma = args['gamma']
		self.lamda = args['lamda']

	def train(self, train_step, name, value_only=False):
		for _ in range(train_step): # train step
			self._episodes += 1

			self.actor.eval()
			self.critic.eval()

			replay_buffer, score = self.get_replay_buffer(self.gamma, self.env[0])

			if self.maxscore < score:
				print("saved at {}".format(name))
				self.maxscore = score
				torch.save(self.get_ckpt(), name)

			states, actions = replay_buffer.get_tensor()

			old_policy, _ = self.actor.evaluate(states, actions, detach=True)
			old_values = self.critic.get_values(states).detach()

			returns, advants = replay_buffer.get_gae(old_values, self.gamma, self.lamda) # gamma
			
			criterion = torch.nn.MSELoss()
			n = len(states)
			batch_size = self.batch_size
			
			self.actor.train()
			self.critic.train()
	
			# TODO : to GPU
			actor_loss_total, critic_loss_total, step = 0, 0, 0
			for epoch in range(1):
				arr = torch.randperm(n)
				for i in range(n // batch_size):
					batch_index = arr[batch_size * i : batch_size * (i+1)]
					states_samples = states[batch_index]
					returns_samples = returns[batch_index]
					advants_samples = advants[batch_index]
					actions_samples = actions[batch_index]
					oldvalues_samples = old_values[batch_index]
					oldpolicy_samples = old_policy[batch_index]

					# surrogate function
					new_policy, entropy = self.actor.evaluate(states_samples, actions_samples)
					ratio = torch.exp(new_policy - oldpolicy_samples)
					loss = ratio * advants_samples

					# clip
					values = self.critic.get_values(states_samples)
					#clipped_values = oldvalues_samples + torch.clamp(values - oldvalues_samples, -0.2, 0.2) # clip param
					#critic_loss1 = criterion(clipped_values, returns_samples)
					critic_loss2 = criterion(values, returns_samples)
					#critic_loss = torch.max(critic_loss1, critic_loss2).mean()

					clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) # clip param
					clipped_loss = clipped_ratio * advants_samples
					actor_loss = -torch.min(loss, clipped_loss).mean()

					# merge loss function
					if value_only: loss = 0.5 * critic_loss2
					else: loss = actor_loss + 0.5 * critic_loss2# - 0.01*entropy.mean()

					# profile
					actor_loss_total += actor_loss.data
					critic_loss_total += critic_loss2.data
					step += 1

					self.actor.zero_grad()
					self.critic.zero_grad()
					loss.backward()
					self.actor.step()
					self.critic.step()

#					self.actor.apply_loss(loss, retain_graph=True)
#					self.critic.apply_loss(loss, retain_graph=False)
		self.actor.eval()
		self.critic.eval()
		print("actor loss avg: {}, critic loss avg: {}".format(actor_loss_total / step, critic_loss_total / step), flush=True)

class VanilaAgent(AgentInterface):
	def __init__(self, env, actor, critic, args, render):
		super(VanilaAgent, self).__init__(env, actor, critic, args, render)
		assert('gamma' in args)
		self.gamma = args['gamma']
		
	def train(self, train_step):
		for _ in range(train_step): # train step
			self.actor.mode_eval()
			self.critic.mode_eval()

			replay_buffer = self.get_replay_buffer()
			states, actions = replay_buffer.get_tensor()

			returns = replay_buffer.get_returns(self.gamma) # gamma
		
			self.actor.mode_train()
			self.critic.mode_train()
			
			log_policy = self.actor.log_policy(states, actions)
			returns = returns.unsqueeze(1)
			loss = -(returns * log_policy).mean()
			
			self.actor.apply_loss(loss)
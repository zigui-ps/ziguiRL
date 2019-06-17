import gym
import torch
from functools import reduce

class GymInterface():
    def __init__(self):
        pass

    # next state + reward
    
    # input: action, state
    # action(torch.Tensor(array: shape = action_space))
    # state(torch.Tensor(array: shape = observation_space)

    # output
    # next_state, reward, done, timeout, info
    # next_state(torch.Tensor(array: shape = observation_space))
    # reward(float)
    # done(boolean)
    # info(??)
    def step(self, action, state = None):
        assert(False)

    # only rendering
    def render(self):
        assert(False)

    # return: size(tuple int)
    def observation_size(self):
        assert(False)

    # return: size(tuple BOX)
    def observation_space(self):
        assert(False)

    # return: size(tuple int)
    def action_size(self):
        assert(False)

    # return: size(tuple BOX)
    def action_space(self):
        assert(False)

    # for test
    # return: action(torch.Tensor)
    def random_action(self):
        assert(False)

class PythonGym():
    def __init__(self, name, limit = -1):
        self.env = gym.make(name)
        self.name = name
        self.last_state = self.env.reset()
        self._step = 0
        self._limit = limit

    def reset(self):
        self.last_state = torch.Tensor(self.env.reset())
        return torch.Tensor(self.last_state)

    def step(self, action):
        # setting state is ignored
        state, reward, done, info = self.env.step(action)
        self.last_state = torch.Tensor(state)
        self._step += 1
        return self.last_state, reward, done, self._step == self._limit, info

    def render(self):
        self.env.render()

    def observation_size(self):
        return self.env.observation_space.shape
    
    def observation_space(self):
        return self.env.observation_space

    def action_size(self):
        return self.env.action_space.shape

    def action_space(self):
        return self.env.action_space

    def random_action(self):
        return torch.Tensor(self.env.action_space.sample())

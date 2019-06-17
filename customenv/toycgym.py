from ctypes import *

BALL = 10
class State(Structure):
    _fields_ = [("s", c_double * BALL), ("v", c_double * BALL), ("a", c_double * BALL)]
    
class Action(Structure):
    _fields_ = [("a", c_double)]

class Result(Structure):
    _fields_ = [("state", State), ("reward", c_double), ("done", c_bool), ("time_limit", c_bool)]
    
class ToyCGym():
    def __init__(self):
        env = cdll.LoadLibrary("./customenv/toycgym.so")
        self._reset = env._Z5resetv
        self._step = env._Z4step6Action
        self._render = env._Z6renderv
        self._reset.argtypes, self._reset.restype = [], State
        self._step.argtypes, self._step.restype = [Action], Result
        self._render.argtypes, self._render.restype = [], None

    def state_modifier(self, cstate):
        state = []
        for i in range(BALL):
            state.append(cstate.s[i])
            state.append(cstate.v[i])
            state.append(cstate.a[i])
        return torch.Tensor(state)

    def reset(self):
        return self.state_modifier(self._reset())

    def step(self, action):
        result = self._step(Action(action[0]))
        return self.state_modifier(result.state), result.reward, result.done, result.time_limit, None

    def render(self):
        self._render()

    def observation_size(self):
        return [BALL * 3]
    
    def action_size(self):
        return [1]

    def observation_space(self):
        assert(False)
    
    def action_space(self):
        assert(False)


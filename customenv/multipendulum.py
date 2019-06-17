import sys
import torch
from ctypes import *

dofs = 7
forces = 5

def get_args():
    argc = len(sys.argv)
    argv = (POINTER(c_char) * (argc + 1))()
    for i, arg in enumerate(sys.argv):
        enc_arg = arg.encode('utf-8')
        argv[i] = create_string_buffer(enc_arg)
    return argc, argv

class State(Structure):
    _fields_ = [("current", c_double * dofs), ("prev", c_double * dofs)]
    
class Action(Structure):
    _fields_ = [("forces", c_double * forces)]

class Result(Structure):
    _fields_ = [("state", State), ("reward", c_double), ("done", c_bool), ("time_limit", c_bool)]
    
class MultiPendulum():
    def __init__(self):
        env = cdll.LoadLibrary("./customenv/build/libMultiPendulum.so")
        self.name = "MultiPendulum"

        self._reset = env._Z5resetv
        self._step = env._Z4step6Action
        self._render = env._Z6renderv
        self._init = env._Z4initiPPc
        
        self._reset.argtypes, self._reset.restype = [], State
        self._step.argtypes, self._step.restype = [Action], Result
        self._render.argtypes, self._render.restype = [], None
        self._init.argtypes, self._init.restype = [c_int, POINTER(POINTER(c_char))], None

        argc, argv = get_args()
        self._init(argc, argv)

    def state_modifier(self, cstate):
        state = []
        for i in range(dofs):
            state.append(cstate.current[i])
            state.append(cstate.prev[i])
        return torch.Tensor(state)

    def reset(self):
        return self.state_modifier(self._reset())

    def step(self, action):
        assert(len(action) == forces)
        
        act = (c_double * forces)()
        for i in range(forces): act[i] = action[i]

        result = self._step(Action(act))
        return self.state_modifier(result.state), result.reward, result.done, result.time_limit, None

    def render(self):
        self._render()

    def observation_size(self):
        return [dofs * 2]
    
    def action_size(self):
        return [forces]

    def observation_space(self):
        assert(False)
    
    def action_space(self):
        assert(False)


import torch

class StateModifierInterface():
    def __init__(self):
        pass

    def apply(self, state):
        return self.modify(state)

    def modify(self, state):
        return state

class ClassicModifier(StateModifierInterface):
    def __init__(self):
        self._n = 0

    def apply(self, state):
        self._n += 1
        if self._n == 1:
            self._mean = state
            self._std = torch.zeros(len(state))
        else:
            prev_mean = self._mean.clone()
            self._mean = prev_mean + (state - prev_mean) / self._n
            self._std = self._std + (state - prev_mean) * (state - self._mean)

        return self.modify(state)

    def modify(self, state):
        if self._n == 1: norm = torch.zeros(state.size())
        else: norm = (state - self._mean) / (1e-8 + torch.sqrt(torch.div(self._std, self._n-1)))
        return torch.clamp(norm, -5, 5)

class DefaultModifier(StateModifierInterface):
    def __init__(self):
        self._n = 0

    def apply(self, state):
        return self.modify(state)

    def modify(self, state):
        return state

import torch

class DefaultModifier():
    def __init__(self):
        pass

    def apply(self, state):
        return self.modify(state)

    def modify(self, state):
        return state

    def set_ckpt(self, ckpt):
        pass

    def get_ckpt(self):
        return {}

class ClassicModifier():
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
        if self._n == 0: return state
        elif self._n == 1: norm = torch.zeros(state.size())
        else: norm = (state - self._mean) / (1e-8 + torch.sqrt(torch.div(self._std, self._n-1)))
        return torch.clamp(norm, -5, 5)

    def set_ckpt(self, ckpt):
        self._n = ckpt['mod_n']
        self._mean = ckpt['mod_mean']
        self._std = ckpt['mod_std']

    def get_ckpt(self):
        return {'mod_n' : self._n, 'mod_mean' : self._mean, 'mod_std' : self._std}


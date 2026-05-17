import numpy as np

class LewisGame:
    def __init__(self, n_states=3, n_signals=3, n_actions=None, lr=0.1, seed=None):
        self.rng = np.random.default_rng(seed)
        self.S = n_states
        self.K = n_signals
        self.A = n_actions if n_actions is not None else n_states
        self.sender = np.ones((self.S, self.K)) / self.K
        self.receiver = np.ones((self.K, self.A)) / self.A
        self.lr = lr

    def step(self, state):
        signal = self.rng.choice(self.K, p=self.sender[state])
        action = self.rng.choice(self.A, p=self.receiver[signal])
        reward = 1.0 if action == state else 0.0
        return signal, action, reward

    def update(self, state, signal, action, reward):
        if reward > 0:
            self.sender[state] += self.lr * (np.eye(self.K)[signal] - self.sender[state])
            self.receiver[signal] += self.lr * (np.eye(self.A)[action] - self.receiver[signal])
        else:
            self.sender[state] -= self.lr * (np.eye(self.K)[signal] - self.sender[state]) * 0.1
            self.receiver[signal] -= self.lr * (np.eye(self.A)[action] - self.receiver[signal]) * 0.1
        self.sender[state] = np.clip(self.sender[state], 1e-6, None)
        self.sender[state] /= self.sender[state].sum()
        self.receiver[signal] = np.clip(self.receiver[signal], 1e-6, None)
        self.receiver[signal] /= self.receiver[signal].sum()

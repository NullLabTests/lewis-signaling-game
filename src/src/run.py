# force non-GUI backend for matplotlib to avoid GTK issues
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from game import LewisGame

def train(steps=5000, seed=0):
    game = LewisGame(n_states=3, n_signals=3, lr=0.2, seed=seed)
    rewards = []
    window = 100
    accs = []
    for t in range(steps):
        state = game.rng.integers(game.S)
        sgn, act, r = game.step(state)
        game.update(state, sgn, act, r)
        rewards.append(r)
        if t >= window:
            accs.append(np.mean(rewards[-window:]))
        else:
            accs.append(np.mean(rewards))
    return game, rewards, accs

if __name__ == "__main__":
    game, rewards, accs = train(steps=5000, seed=1)
    plt.plot(accs)
    plt.xlabel("Step")
    plt.ylabel("Moving average reward")
    plt.title("Lewis Signaling Game learning curve")
    plt.grid(True)
    out = "learning_curve.png"
    plt.savefig(out)
    print(f"Saved {out}")

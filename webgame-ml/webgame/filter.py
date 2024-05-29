from typing import *
import numpy as np


class BayesFilter:
    """
    A discrete Bayes filter for localization.
    """

    def __init__(self, size: int):
        self.prior = np.ones([size, size]) / size**2

    def localize(self, obs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Given an agent's observations, returns the new pre
        """
        return self.prior
    
if __name__ == "__main__":
    from webgame.envs import GameEnv
    from matplotlib import pyplot as plt # type: ignore

    env = GameEnv(visualize=True)
    action_space = env.action_space("pursuer") # Same for both agents
    b_filter = BayesFilter(env.level_size)
    for _ in range(1000):
        actions = {}
        for agent in env.agents:
            actions[agent] = action_space.sample()
        obs_all = env.step(actions)[0]

        obs = obs_all["pursuer"]
        probs = b_filter.localize(obs)
        plt.imshow(probs)
        plt.show()

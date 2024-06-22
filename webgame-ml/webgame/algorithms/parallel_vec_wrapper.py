import pettingzoo as pz  # type: ignore
from typing import *
import gymnasium as gym
import numpy as np


class ParallelVecWrapper(pz.ParallelEnv):
    """
    Wraps multiple instances of `ParallelEnv`s for vectorized multi-agent training.

    Similar to `ParallelEnv`, a dict keyed by agent names is returned, and similar to vectorized envs, each agent
    receives a batch of data from each sub environment.

    Assumes the number of agents is constant (e.g. agents neither die nor spawn).
    """

    def __init__(self, env_fns: Iterable[Callable[[], pz.ParallelEnv]]):
        self.envs = [env_fn() for env_fn in env_fns]
        self.agents = self.envs[0].agents

    def action_space(self, agent: str) -> gym.Space:
        return self.envs[0].action_space(agent)

    def observation_space(self, agent: str) -> gym.Space:
        return self.envs[0].observation_space(agent)

    def step(self, actions: Dict[str, np.ndarray]) -> tuple[
        dict,
        dict[Any, List[float]],
        dict[Any, List[bool]],
        dict[Any, List[bool]],
        dict[Any, List[dict]],
    ]:
        agent_obs_all: Dict[str, List[Any]] = {agent: [] for agent in self.agents}
        agent_rew_all: Dict[str, List[float]] = {agent: [] for agent in self.agents}
        agent_done_all: Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        agent_trunc_all: Dict[str, List[bool]] = {agent: [] for agent in self.agents}
        agent_info_all: Dict[str, List[dict]] = {agent: [] for agent in self.agents}
        for i, env in enumerate(self.envs):
            all_actions = {}
            for agent, action in actions.items():
                all_actions[agent] = action[i]
            obs, rew, done, trunc, info = env.step(all_actions)
            if done[self.agents[0]] or trunc[self.agents[0]]:
                obs, info = env.reset()
            for agent in self.agents:
                agent_obs_all[agent].append(obs[agent])
                agent_rew_all[agent].append(rew[agent])
                agent_done_all[agent].append(done[agent])
                agent_trunc_all[agent].append(trunc[agent])
                agent_info_all[agent].append(info[agent])
        agent_obs_all = {
            agent: ParallelVecWrapper._stack_obs(
                obs, self.observation_space(self.agents[0])
            )
            for agent, obs in agent_obs_all.items()
        }
        return (
            agent_obs_all,
            agent_rew_all,
            agent_done_all,
            agent_trunc_all,
            agent_info_all,
        )

    def reset(self, *args) -> tuple[dict, dict[Any, List[dict]]]:
        agent_obs_all: Dict[str, List[Any]] = {agent: [] for agent in self.agents}
        agent_info_all: Dict[str, List[dict]] = {agent: [] for agent in self.agents}
        for env in self.envs:
            obs, info = env.reset(*args)
            for agent in env.agents:
                agent_obs_all[agent].append(obs[agent])
                agent_info_all[agent].append(info[agent])
        agent_obs_all = {
            agent: ParallelVecWrapper._stack_obs(
                obs, self.observation_space(self.agents[0])
            )
            for agent, obs in agent_obs_all.items()
        }
        return agent_obs_all, agent_info_all

    @staticmethod
    def _stack_obs(obs: Any, space: gym.Space) -> Any:
        """
        Stacks observations, if applicable.

        If the obs is a list of `Box`es, they are stacked. If the obs is a `Tuple`, this method is called on each of the
        sub observations (i.e. if obs is originally a list of tuples, a tuple of lists is returned, possibly stacked).
        """
        if isinstance(space, gym.spaces.Box):
            return np.stack(obs)
        if isinstance(space, gym.spaces.Tuple):
            processed_obs: List[List[Any]] = [[] for _ in range(len(space.spaces))]
            for env_obs in obs:
                for i in range(len(space.spaces)):
                    processed_obs[i].append(env_obs[i])
            for i, sub_space in enumerate(space.spaces):
                processed_obs[i] = ParallelVecWrapper._stack_obs(
                    processed_obs[i], sub_space
                )
            return processed_obs
        return obs

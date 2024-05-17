from typing import *
import gymnasium as gym
import pettingzoo  # type: ignore
# import rerun as rr  # type: ignore
from tqdm import tqdm
from webgame_rust import AgentState, GameWrapper, GameState
import numpy as np
import functools

# The maximum number of object vectors supported by the environment.
MAX_OBJS = 16
# The dimension of each object vector.
OBJ_DIM = 2

# The world space size of a grid cell
CELL_SIZE = 25

class GameEnv(pettingzoo.ParallelEnv):
    """
    An environment that wraps an instance of our game.

    Agents: pursuer, player

    Observation Space: A tuple, where the first item is a list of vectors representing visible objects in the game from
    an agent's POV, and the second item is a 2D map showing where walls are.

    Action Space: Discrete, check the `AgentAction` enum for a complete list.

    Args:
        visualize: If we should log visuals to Rerun.
    """

    def __init__(self, visualize: bool = False):
        self.game = GameWrapper(visualize)
        self.game_state: Optional[GameState] = None
        self.last_obs: Optional[Mapping[str, tuple[np.ndarray, np.ndarray]]] = None
        self.possible_agents = ["player", "pursuer"]
        self.agents = self.possible_agents[:]
        self.level_size = 8

    def step(self, actions: Mapping[str, int]) -> tuple[
        Mapping[str, tuple[np.ndarray, np.ndarray]],
        Mapping[str, float],
        Mapping[str, bool],
        Mapping[str, bool],
        Mapping[str, None],
    ]:
        all_actions = []
        for agent in ["player", "pursuer"]:
            all_actions.append(actions[agent])
        self.game_state = self.game.step(all_actions[0], all_actions[1])
        assert self.game_state
        obs = game_state_to_obs(self.game_state)
        self.last_obs = obs
        rewards = {
            "player": 0.0,
            "pursuer": 0.0,
        }
        dones = {
            "player": False,
            "pursuer": False,
        }
        truncs = {
            "player": False,
            "pursuer": False,
        }
        infos = {
            "player": None,
            "pursuer": None,
        }
        return (obs, rewards, dones, truncs, infos)

    def reset(
        self, *args
    ) -> tuple[Mapping[str, tuple[np.ndarray, np.ndarray]], dict[str, None]]:
        self.game_state = self.game.reset()
        assert self.game_state
        obs = game_state_to_obs(self.game_state)
        infos = {
            "player": None,
            "pursuer": None,
        }
        return (obs, infos)

    @functools.lru_cache(maxsize=None)
    def action_space(self, _: str) -> gym.Space:
        return gym.spaces.Discrete(10)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _: str) -> gym.Space:
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(0, 1, (MAX_OBJS, OBJ_DIM)),
                gym.spaces.Box(0, 1, (self.level_size, self.level_size)),
            )
        )


def game_state_to_obs(
    game_state: GameState,
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    """
    Converts the game state to our expected observations.
    """
    return {
        "player": agent_state_to_obs(game_state.player, game_state),
        "pursuer": agent_state_to_obs(game_state.pursuer, game_state),
    }


def agent_state_to_obs(
    agent_state: AgentState, game_state: GameState
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates observations for an agent.
    """
    obs_vecs = np.zeros([MAX_OBJS, OBJ_DIM], dtype=float)
    for i, e in enumerate(agent_state.observing):
        obj = game_state.objects[e]
        obj_features = np.zeros([OBJ_DIM])
        obj_features[0] = obj.pos.x / (game_state.level_size * CELL_SIZE)
        obj_features[1] = obj.pos.y / (game_state.level_size * CELL_SIZE)
        obs_vecs[i] = obj_features
    walls = np.array(game_state.walls, dtype=float).reshape(
        (game_state.level_size, game_state.level_size)
    )
    return (obs_vecs, walls)


if __name__ == "__main__":
    env = GameEnv(visualize=True)
    # env.reset()
    for _ in tqdm(range(1000)):
        env.step(
            {
                "player": env.action_space("player").sample(),
                "pursuer": env.action_space("pursuer").sample(),
            }
        )

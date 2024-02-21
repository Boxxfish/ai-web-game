from typing import *
import gymnasium as gym
import pettingzoo  # type: ignore
import rerun as rr  # type: ignore
from webgame_rust import AgentState, GameWrapper, GameState
import numpy as np
import functools

# The maximum number of object vectors supported by the environment.
MAX_OBJS = 16
# The dimension of each object vector.
OBJ_DIM = 2


class GameEnv(pettingzoo.ParallelEnv):
    """
    An environment that wraps an instance of our game.

    Agents: pursuer, player

    Observation Space: A tuple, where the first item is a list of vectors representing visible objects in the game from
    an agent's POV, and the second item is a 2D map showing where walls are.

    Action Space: Discrete, check the `AgentAction` enum for a complete list.
    """

    def __init__(self):
        self.game = GameWrapper()
        self.game_state: Optional[GameState] = None
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
        return gym.spaces.Discrete(9)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, _: str) -> gym.Space:
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(0, 1, (MAX_OBJS, OBJ_DIM)),
                gym.spaces.Box(0, 1, (self.level_size, self.level_size)),
            )
        )

    def log_render(self):
        """
        When called, logs a visual to the current recording.
        This allows both saving episodes and watching them live.
        """
        if self.game_state:
            walls = 1 - np.array(self.game_state.walls, dtype=float).reshape(
                (self.level_size, self.level_size)
            )
            rr.log("game/walls", rr.Image(walls))
            player_pos = (
                np.array(
                    [
                        self.game_state.player.pos.x,
                        self.level_size * 25 - self.game_state.player.pos.y - 25,
                    ]
                )
                / 25
                + 0.5
            )
            rr.log(
                "game/player_pos",
                rr.Points2D([player_pos], radii=0.25, colors=(0.0, 1.0, 0.0)),
            )
            pursuer_pos = (
                np.array(
                    [
                        self.game_state.pursuer.pos.x,
                        self.level_size * 25 - self.game_state.pursuer.pos.y - 25,
                    ]
                )
                / 25
                + 0.5
            )
            rr.log(
                "game/pursuer_pos",
                rr.Points2D([pursuer_pos], radii=0.25, colors=(1.0, 0.0, 0.0)),
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
    # TODO: Replace this with visible objects
    obs_vecs = np.zeros([MAX_OBJS, OBJ_DIM], dtype=float)
    walls = np.array(game_state.walls, dtype=float).reshape(
        (game_state.level_size, game_state.level_size)
    )
    return (obs_vecs, walls)


if __name__ == "__main__":
    rr.init("Env Test")
    rr.connect()

    env = GameEnv()
    env.reset()
    for _ in range(1000):
        env.step(
            {
                "player": env.action_space("player").sample(),
                "pursuer": env.action_space("pursuer").sample(),
            }
        )
        env.log_render()

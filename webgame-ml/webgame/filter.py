from typing import *
import numpy as np
from scipy import signal  # type: ignore

import torch
from torch.distributions import Categorical
from safetensors.torch import load_model
from webgame_rust import AgentState, GameState

from webgame.common import convert_obs, explore_policy, pos_to_grid, process_obs

import gymnasium as gym
from torch import Tensor, nn
from webgame.models import MeasureModel, PolicyNet


class BayesFilter:
    """
    A discrete Bayes filter for localization.
    """

    def __init__(
        self,
        size: int,
        cell_size: float,
        update_fn: Callable[
            [
                Tuple[np.ndarray, np.ndarray, np.ndarray],
                bool,
                GameState,
                AgentState,
                int,
                float,
                bool,
            ],
            np.ndarray,
        ],
        use_objs: bool,
        is_pursuer: bool,
        lkhd_min: float = 0.0,
    ):
        self.size = size
        self.cell_size = cell_size
        self.belief = np.ones([size, size]) / size**2
        self.update_fn = update_fn
        self.use_objs = use_objs
        self.is_pursuer = is_pursuer
        self.lkhd_min = lkhd_min

    def localize(
        self,
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        game_state: GameState,
        agent_state: AgentState,
    ) -> np.ndarray:
        """
        Given an agent's observations, returns the new location probabilities.
        """
        walls = np.array(game_state.walls).astype(float)
        walls = walls.reshape([game_state.level_size, game_state.level_size])
        self.belief = self.predict(self.belief, walls)
        lkhd = self.update_fn(
            obs,
            self.use_objs,
            game_state,
            agent_state,
            self.size,
            self.cell_size,
            self.is_pursuer,
        )
        lkhd = lkhd * (1 - self.lkhd_min) + self.lkhd_min
        self.belief = lkhd * self.belief
        self.belief = self.belief / self.belief.sum()
        return self.belief

    def predict(self, belief: np.ndarray, walls: np.ndarray) -> np.ndarray:
        kernel = np.array([[0.25, 1, 0.25], [1, 1, 1], [0.25, 1, 0.25]])
        belief = signal.convolve2d(belief, kernel, mode="same")
        denom = signal.convolve2d(1.0 - walls, kernel, mode="same") + 0.001
        return belief / denom


def manual_update(
    obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    use_objs: bool,
    game_state: GameState,
    agent_state: AgentState,
    size: int,
    cell_size: float,
    is_pursuer: bool,
) -> np.ndarray:
    assert not use_objs, "Not compatible with objects"
    # Check whether agent can see the player
    other_agent = ["pursuer", "player"][int(is_pursuer)]
    other_e, other_obs = list(
        filter(lambda t: t[1].obj_type == other_agent, game_state.objects.items())
    )[0]
    player_vis_grid = None
    if other_e in agent_state.observing:
        player_vis_grid = pos_to_grid(other_obs.pos.x, other_obs.pos.y, size, cell_size)

    obs_grid = np.array(game_state.walls).reshape(
        [game_state.level_size, game_state.level_size]
    )
    lkhd = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            grid_lkhd = 1 - obs_grid[y][x]
            agent_lkhd = 1.0
            if player_vis_grid is not None:
                if player_vis_grid != (x, y):
                    agent_lkhd = 0.0
            else:
                # Cells within vision have 0% chance of agent being there
                agent_lkhd = (
                    1.0 - agent_state.visible_cells[y * game_state.level_size + x]
                )
                # All other cells are equally probable
                agent_lkhd = agent_lkhd / (size**2 - sum(agent_state.visible_cells))

            lkhd[y][x] = grid_lkhd * agent_lkhd
    return lkhd


def model_update(
    model: nn.Module,
) -> Callable[
    [
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        bool,
        GameState,
        AgentState,
        int,
        float,
        bool,
    ],
    np.ndarray,
]:
    @torch.no_grad()
    def model_update_(
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
        use_objs: bool,
        game_state: GameState,
        agent_state: AgentState,
        size: int,
        cell_size: float,
        is_pursuer: bool,
    ) -> np.ndarray:
        lkhd = (
            model(
                torch.from_numpy(obs[0]).float(),
                torch.from_numpy(obs[1]).float() if use_objs else None,
                torch.from_numpy(obs[2]).float() if use_objs else None,
            )
            .squeeze(0)
            .numpy()
        )
        return lkhd

    return model_update_


def gt_update(
    obs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    use_objs: bool,
    game_state: GameState,
    agent_state: AgentState,
    size: int,
    cell_size: float,
    is_pursuer: bool,
) -> np.ndarray:
    player_pos = game_state.player.pos
    grid_pos = pos_to_grid(player_pos.x, player_pos.y, game_state.level_size, CELL_SIZE)
    lkhd = np.zeros([size, size])
    lkhd[grid_pos[1], grid_pos[0]] = 1
    kernel = np.array([[0.1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 0.1]])
    lkhd = signal.convolve2d(lkhd, kernel, mode="same")
    return lkhd


def replace_extra_channel(obs: Tuple[Tensor, Tensor, Tensor], channel: Tensor):
    """
    Replaces the extra channel.
    """
    zeroed = obs[0]
    zeroed[:, -1] = channel
    return (zeroed.numpy(), obs[1].numpy(), obs[2].numpy())


if __name__ == "__main__":
    from webgame.envs import GameEnv, CELL_SIZE, MAX_OBJS, OBJ_DIM
    import rerun as rr  # type: ignore
    import random
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--pursuer-chkpt", type=str, default=None)
    parser.add_argument("--player-chkpt", type=str, default=None)
    parser.add_argument("--use-pos", action="store_true")
    parser.add_argument("--use-objs", action="store_true")
    parser.add_argument("--use-gt", action="store_true")
    parser.add_argument("--wall-prob", type=float, default=0.1)
    parser.add_argument("--lkhd-min", type=float, default=0.0)
    parser.add_argument("--insert-visible-cells", default=False, action="store_true")
    parser.add_argument("--start-gt", default=False, action="store_true")
    parser.add_argument(
        "--player-sees-visible-cells", default=False, action="store_true"
    )
    args = parser.parse_args()

    def heuristic_policy(
        agent: str, env: GameEnv, obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> int:
        if random.random() < 0.1:
            action = env.action_space(agent).sample()
        else:
            assert env.game_state
            action = explore_policy(env.game_state, agent == "pursuer")
        return action

    def model_policy(
        chkpt_path: str,
        action_count: int,
        use_pos: bool,
        use_objs: bool,
    ) -> Callable[[str, GameEnv, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
        channels = 9
        if args.player_sees_visible_cells:
            channels = 10
        p_net = PolicyNet(
            channels,
            8,
            action_count,
            use_pos,
            (MAX_OBJS, OBJ_DIM) if use_objs else None,
        )
        load_model(p_net, chkpt_path)

        def policy(
            agent: str,
            env: GameEnv,
            obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> int:
            action_probs = p_net(*obs).squeeze(0)
            action = Categorical(logits=action_probs).sample().item()
            return action

        return policy

    recording_id = "filter_test-" + str(random.randint(0, 10000))
    rr.init(application_id="Pursuer", recording_id=recording_id)
    rr.connect()

    env = GameEnv(
        wall_prob=args.wall_prob,
        use_objs=args.use_objs,
        visualize=True,
        recording_id=recording_id,
        player_sees_visible_cells=args.player_sees_visible_cells,
    )
    obs_ = env.reset()[0]
    obs = {agent: convert_obs(obs_[agent], True) for agent in env.agents}

    action_space = env.action_space("pursuer")  # Same for both agents
    assert isinstance(action_space, gym.spaces.Discrete)
    assert env.game_state is not None

    # Set up filter
    if args.checkpoint:
        model = MeasureModel(9, env.game_state.level_size, args.use_pos)
        load_model(model, args.checkpoint)
        update_fn = model_update(model)
    elif args.use_gt:
        update_fn = gt_update
    else:
        update_fn = manual_update
    b_filter = BayesFilter(
        env.game_state.level_size,
        CELL_SIZE,
        update_fn,
        args.use_objs,
        True,
        args.lkhd_min,
    )
    if args.start_gt:
        b_filter.belief = np.zeros(b_filter.belief.shape)
        play_pos = env.game_state.player.pos
        x, y = pos_to_grid(play_pos.x, play_pos.y, env.game_state.level_size, CELL_SIZE)
        b_filter.belief[y, x] = 1

    # Set up policies
    policies = {}
    chkpts = {"pursuer": args.pursuer_chkpt, "player": args.player_chkpt}
    for agent in env.agents:
        if chkpts[agent]:
            policies[agent] = model_policy(
                chkpts[agent], int(action_space.n), args.use_pos, args.use_objs
            )
        else:
            policies[agent] = heuristic_policy

    for _ in range(100):
        actions = {}
        for agent in env.agents:
            action = policies[agent](agent, env, obs[agent])
            actions[agent] = action
        obs = {
            agent: convert_obs(env.step(actions)[0][agent], True)
            for agent in env.agents
        }

        game_state = env.game_state
        assert game_state is not None
        agent_state = game_state.pursuer
        extra_channel = torch.zeros(
            [game_state.level_size, game_state.level_size], dtype=torch.float
        )
        if args.insert_visible_cells:
            visible_cells = agent_state.visible_cells
            extra_channel = torch.tensor(visible_cells, dtype=torch.float).reshape(
                [game_state.level_size, game_state.level_size]
            )
        filter_obs = replace_extra_channel(obs["pursuer"], extra_channel)
        lkhd = update_fn(
            filter_obs,
            False,
            game_state,
            agent_state,
            game_state.level_size,
            CELL_SIZE,
            True,
        )
        manual_lkhd = manual_update(
            filter_obs,
            False,
            game_state,
            agent_state,
            game_state.level_size,
            CELL_SIZE,
            True,
        )
        probs = b_filter.localize(filter_obs, game_state, agent_state)
        rr.log("filter/belief", rr.Tensor(probs), timeless=False)
        rr.log("filter/measurement_likelihood", rr.Tensor(lkhd), timeless=False)
        rr.log(
            "filter/manual_measurement_likelihood",
            rr.Tensor(manual_lkhd),
            timeless=False,
        )
        rr.log("filter/filter_obs", rr.Tensor(filter_obs[0]), timeless=False)

"""
Script for benchmarking filters.
"""

from typing import *

from safetensors.torch import load_model
from tqdm import tqdm

from webgame.common import explore_policy, pos_to_grid, process_obs
from webgame.envs import CELL_SIZE, GameEnv
from webgame.filter import BayesFilter, manual_update, model_update
from webgame.train_filter import MeasureModel
import random
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use-objs", default=False, action="store_true")
    parser.add_argument("--use-pos", default=False, action="store_true")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--run-steps", type=int, default=100)
    args = parser.parse_args()

    env = GameEnv(use_objs=args.use_objs)
    env.reset()
    assert env.game_state is not None

    action_space = env.action_space("pursuer")  # Same for both agents
    if args.checkpoint:
        model = MeasureModel(8, env.game_state.level_size, args.use_pos)
        model.eval()
        load_model(model, args.checkpoint)
        update_fn = model_update(model)
    else:
        update_fn = manual_update
    correct_preds = 0
    exp_accuracy = 0.0
    for run in tqdm(range(args.num_runs)):
        env.reset()
        assert env.game_state is not None
        b_filter = BayesFilter(
            env.game_state.level_size, CELL_SIZE, update_fn, use_objs=args.use_objs
        )
        for step in range(args.run_steps):
            actions = {}
            for agent in env.agents:
                if random.random() < 0.1:
                    action = action_space.sample()
                else:
                    action = explore_policy(env.game_state, agent == "pursuer")
                actions[agent] = action
            obs = process_obs(env.step(actions)[0]["pursuer"])

            game_state = env.game_state
            assert game_state is not None
            agent_state = game_state.pursuer
            lkhd = update_fn(
                obs,
                args.use_objs,
                game_state,
                agent_state,
                game_state.level_size,
                CELL_SIZE,
            )
            probs = b_filter.localize(obs, game_state, agent_state)
            probs_flattened = probs.flatten()

            player_pos = env.game_state.player.pos
            gold_tile = pos_to_grid(
                player_pos.x, player_pos.y, env.game_state.level_size, CELL_SIZE
            )
            gold_tile_idx = gold_tile[0] + gold_tile[1] * env.game_state.level_size

            if probs_flattened.argmax() == gold_tile_idx:
                correct_preds += 1
            exp_accuracy += float(probs_flattened[gold_tile_idx])

    print("Accuracy:", correct_preds / (args.num_runs * args.run_steps))
    print("Expected Accuracy:", exp_accuracy / (args.num_runs * args.run_steps))

# Pursuer

**Note: This repository is in semi-archived mode. Feel free to fork and make any changes, but aside from major issues no changes will be made to this repository. Thanks!**

[![Video](https://img.youtube.com/vi/54bOSzWGBEw/maxresdefault.jpg)](https://youtu.be/54bOSzWGBEw)

This repo contains both the game and training code for the AI demo "Pursuer". The design document can be found in the `/docs` folder.

## Running the Game

Everything related to game code can be found in the `webgame-game` directory.

To build and play the game, run the following to generate a release build:

```bash
cd webgame-game
cargo run --release
```

If you're working on the game, you'll want to use dynamic linking, so it only takes a couple seconds to recompile
whenever you make a change. Use this instead:

```bash
cd webgame-game
cargo run --features bevy/dynamic_linking
```

## Running RL Experiments

Everything related to ML can be found in the `webgame-ml` directory.

You'll first need to set up a virtual Poetry environment. First, make sure you've downloaded
[Poetry](https://python-poetry.org/). Once you have, create the environment with the following:

```bash
cd webgame-ml
poetry install
poetry shell
```

You'll need to run `poetry shell` whenever you want to activate the environment.

Next, you'll have to build the Rust extension, which allows our Python code to talk to our game. Run the following:

```bash
cd webgame_rust
maturin develop --release
```

Configure wandb by creating a file called `conf.py` under the `webgame` directory with the following:

```
entity = "YOUR_WANDB_NAME"
```

Now, run an experiment:

```bash
mkdir runs
python webgame/train_agents.py
```

If you open your WandB dashboard, you should see a bunch of metrics pop up now. You should also see a file called
`p_net-ITERATION.safetensors` in your `runs` directory. This file contains the weights of our neural network. To update the game's
current checkpoint, rename this file to `p_net.safetensors` and move this file to the `assets` folder under `webgame-game`.

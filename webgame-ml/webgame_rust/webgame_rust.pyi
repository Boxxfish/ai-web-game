from typing import *

class PyVec2:
    """
    Represents a 2D vector.
    """
    x: float
    y: float

class AgentState:
    """
    Contains the state of an agent for a single frame.
    """
    pos: PyVec2
    dir: PyVec2

class GameState:
    """
    Contains the state of the game for a single frame.
    """
    player: AgentState
    pursuer: AgentState
    walls: list[bool]
    level_size: int

class AgentAction:
    """
    Indicates the kind of actions an agent can take.
    """
    NoAction: int
    MoveUp: int
    MoveUpRight: int
    MoveRight: int
    MoveDownRight: int
    MoveDown: int
    MoveDownLeft: int
    MoveLeft: int
    MoveUpLeft: int

class GameWrapper:
    def __init__(self) -> None: ...
    def step(
        self, action_player: AgentAction, action_pursuer: AgentAction
    ) -> GameState:
        """
        Runs one step of the game, and returns the next state of the game.
        """
        ...
    def reset(self) -> GameState: 
        """
        Resets the game, returning the next state of the game.
        """
        ...

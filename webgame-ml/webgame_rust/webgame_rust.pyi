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

class GameWrapper:
    def __init__(self) -> None: ...
    def step(
        self, action_player: int, action_pursuer: int
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

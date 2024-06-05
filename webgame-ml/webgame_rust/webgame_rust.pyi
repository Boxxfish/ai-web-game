from typing import *

class ObservableObj:
    """
    Describes an observable object.
    """
    pos: PyVec2
    obj_type: str

class NoiseSourceObj:
    """
    Describes a noise source.
    """
    pos: PyVec2
    active_radius: float

class PyVec2:
    """
    Represents a 2D vector.
    """
    x: float
    y: float

class VMData:
    """
    Data on visual markers.
    """
    last_seen: bool
    last_seen_elapsed: Optional[bool]
    last_state: bool
    state_changed: bool

class AgentState:
    """
    Contains the state of an agent for a single frame.
    """
    pos: PyVec2
    dir: PyVec2
    observing: list[int]
    listening: list[int]
    vm_data: Mapping[int, VMData]
    visible_cells: list[bool]

class GameState:
    """
    Contains the state of the game for a single frame.
    """
    player: AgentState
    pursuer: AgentState
    walls: list[bool]
    level_size: int
    objects: Mapping[int, ObservableObj]
    noise_sources: Mapping[int, NoiseSourceObj]

class GameWrapper:
    def __init__(self, visualize: bool, recording_id: Optional[str]) -> None:
        """
        Args:
            visualize: If we should log visuals to Rerun.
        """
        ...
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

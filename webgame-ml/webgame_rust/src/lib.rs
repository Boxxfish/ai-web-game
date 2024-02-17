use bevy::prelude::*;
use pyo3::prelude::*;
use webgame_game::{
    configs::LibCfgPlugin,
    gridworld::{Agent, LevelLayout, NextAction, PlayerAgent, PursuerAgent},
};

/// Represents a 2D vector.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct PyVec2 {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
}

/// Contains the state of an agent for a single frame.
#[pyclass]
#[derive(Debug, Clone)]
pub struct AgentState {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub dir: PyVec2,
}

/// Contains the state of the game for a single frame.
#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    #[pyo3(get)]
    pub player: AgentState,
    #[pyo3(get)]
    pub pursuer: AgentState,
    #[pyo3(get)]
    pub walls: Vec<bool>,
    #[pyo3(get)]
    pub level_size: usize,
}

/// Indicates the kind of actions an agent can take.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub enum AgentAction {
    NoAction = 0,
    MoveUp = 1,
    MoveUpRight = 2,
    MoveRight = 3,
    MoveDownRight = 4,
    MoveDown = 5,
    MoveDownLeft = 6,
    MoveLeft = 7,
    MoveUpLeft = 8,
}

/// Wraps our game in a gym-like interface.
#[pyclass]
pub struct GameWrapper {
    pub app: App,
}

#[pymethods]
impl GameWrapper {
    #[new]
    pub fn new() -> Self {
        let mut app = App::new();
        app.add_plugins(LibCfgPlugin);

        app.finish();
        app.cleanup();
        app.update();

        Self { app }
    }

    pub fn step(&mut self, action_player: AgentAction, action_pursuer: AgentAction) -> GameState {
        set_agent_action::<PlayerAgent>(&mut self.app.world, action_player);
        set_agent_action::<PursuerAgent>(&mut self.app.world, action_pursuer);

        self.app.update();

        self.get_state()
    }

    pub fn reset(&mut self) -> GameState {
        *self = Self::new();
        self.get_state()
    }
}

/// Queries the world for an agent with the provided component and sets the next action.
fn set_agent_action<T: Component>(world: &mut World, action: AgentAction) {
    let mut next_action = world
        .query_filtered::<&mut NextAction, With<T>>()
        .single_mut(world);
    next_action.dir = match action {
        AgentAction::MoveUp => Vec2::Y,
        AgentAction::MoveUpRight => (Vec2::Y + Vec2::X).normalize(),
        AgentAction::MoveRight => Vec2::X,
        AgentAction::MoveDownRight => (-Vec2::Y + Vec2::X).normalize(),
        AgentAction::MoveDown => -Vec2::Y,
        AgentAction::MoveDownLeft => (-Vec2::Y + -Vec2::X).normalize(),
        AgentAction::MoveLeft => -Vec2::X,
        AgentAction::MoveUpLeft => (Vec2::Y + -Vec2::X).normalize(),
        _ => Vec2::ZERO,
    };
}

/// Queries the world for an agent with the provided component and returns an `AgentState`.
fn get_agent_state<T: Component>(world: &mut World) -> AgentState {
    let (agent, xform) = world
        .query_filtered::<(&Agent, &Transform), With<T>>()
        .single(world);
    AgentState {
        pos: PyVec2 {
            x: xform.translation.x,
            y: xform.translation.y,
        },
        dir: PyVec2 {
            x: agent.dir.x,
            y: agent.dir.y,
        },
    }
}

impl GameWrapper {
    fn get_state(&mut self) -> GameState {
        let world = &mut self.app.world;
        let player = get_agent_state::<PlayerAgent>(world);
        let pursuer = get_agent_state::<PursuerAgent>(world);
        let level = world.get_resource::<LevelLayout>().unwrap();
        GameState {
            player,
            pursuer,
            walls: level.walls.clone(),
            level_size: level.size,
        }
    }
}

impl Default for GameWrapper {
    fn default() -> Self {
        Self::new()
    }
}

#[pymodule]
fn webgame_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameWrapper>()?;
    m.add_class::<AgentAction>()?;
    Ok(())
}

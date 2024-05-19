use std::collections::HashMap;

use bevy::{app::AppExit, prelude::*};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::{exceptions::PyValueError, prelude::*};
use webgame_game::{
    configs::{LibCfgPlugin, VisualizerPlugin},
    gridworld::{Agent, LevelLayout, NextAction, PlayerAgent, PursuerAgent},
    observer::{Observable, Observer},
    world_objs::NoiseSource,
};

/// Describes an observable object.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ObservableObject {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub obj_type: String,
}

/// Describes a noise source in the environment.
#[pyclass]
#[derive(Debug, Clone)]
pub struct NoiseSourceObject {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub active_radius: f32,
}

/// Represents a 2D vector.
#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct PyVec2 {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
}

impl From<Vec2> for PyVec2 {
    fn from(value: Vec2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

/// Contains the state of an agent for a single frame.
#[pyclass]
#[derive(Debug, Clone)]
pub struct AgentState {
    #[pyo3(get)]
    pub pos: PyVec2,
    #[pyo3(get)]
    pub dir: PyVec2,
    #[pyo3(get)]
    pub observing: Vec<u64>,
    #[pyo3(get)]
    pub listening: Vec<u64>,
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
    #[pyo3(get)]
    pub objects: HashMap<u64, ObservableObject>,
    #[pyo3(get)]
    pub noise_sources: HashMap<u64, NoiseSourceObject>,
}

/// Indicates the kind of actions an agent can take.
#[derive(Debug, Copy, Clone, TryFromPrimitive, IntoPrimitive, PartialEq, Eq)]
#[repr(u8)]
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
    ToggleObj = 9,
}

impl<'source> FromPyObject<'source> for AgentAction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index: u8 = ob.extract()?;
        Self::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid action"))
    }
}

/// Wraps our game in a gym-like interface.
#[pyclass]
pub struct GameWrapper {
    pub app: App,
    pub visualize: bool,
}

#[pymethods]
impl GameWrapper {
    #[new]
    pub fn new(visualize: bool) -> Self {
        let mut app = App::new();
        app.add_plugins(LibCfgPlugin);

        if visualize {
            app.add_plugins(VisualizerPlugin);
        }

        app.finish();
        app.cleanup();
        app.update();

        Self { app, visualize }
    }

    pub fn step(&mut self, action_player: AgentAction, action_pursuer: AgentAction) -> GameState {
        set_agent_action::<PlayerAgent>(&mut self.app.world, action_player);
        set_agent_action::<PursuerAgent>(&mut self.app.world, action_pursuer);

        self.app.update();

        self.get_state()
    }

    pub fn reset(&mut self) -> GameState {
        self.app.world.send_event(AppExit);
        self.app.run();
        *self = Self::new(self.visualize);
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
    next_action.toggle_objs = action == AgentAction::ToggleObj;
}

/// Queries the world for an agent with the provided component and returns an `AgentState`.
fn get_agent_state<T: Component>(world: &mut World) -> AgentState {
    let (agent, &xform, observer) = world
        .query_filtered::<(&Agent, &GlobalTransform, &Observer), With<T>>()
        .single(world);
    let pos = xform.translation().xy().into();
    let dir = agent.dir.into();
    let observing = observer.observing.iter().map(|e| e.to_bits()).collect();

    let listening = world
        .query::<(Entity, &GlobalTransform, &NoiseSource)>()
        .iter(world)
        .filter(|(_, noise_xform, noise_src)| {
            (xform.translation().xy() - noise_xform.translation().xy()).length_squared()
                <= noise_src.noise_radius
        })
        .map(|(e, _, _)| e.to_bits())
        .collect();
    AgentState {
        pos,
        dir,
        observing,
        listening,
    }
}

impl GameWrapper {
    fn get_state(&mut self) -> GameState {
        let world = &mut self.app.world;
        let player = get_agent_state::<PlayerAgent>(world);
        let pursuer = get_agent_state::<PursuerAgent>(world);

        // Record all observable items
        let mut observables =
            world.query_filtered::<(Entity, &GlobalTransform, Option<&Agent>), With<Observable>>();
        let mut objects = HashMap::new();
        for (e, xform, agent) in observables.iter(world) {
            if agent.is_some() {
                objects.insert(
                    e.to_bits(),
                    ObservableObject {
                        pos: xform.translation().xy().into(),
                        obj_type: "agent".into(),
                    },
                );
            }
        }

        // Record all noise sources
        let mut noise_srcs = world.query::<(Entity, &GlobalTransform, &NoiseSource)>();
        let mut noise_sources = HashMap::new();
        for (e, xform, noise_src) in noise_srcs.iter(world) {
            noise_sources.insert(
                e.to_bits(),
                NoiseSourceObject {
                    pos: xform.translation().xy().into(),
                    active_radius: noise_src.active_radius,
                },
            );
        }

        let level = world.get_resource::<LevelLayout>().unwrap();
        GameState {
            player,
            pursuer,
            walls: level.walls.clone(),
            level_size: level.size,
            objects,
            noise_sources,
        }
    }
}

impl Default for GameWrapper {
    fn default() -> Self {
        Self::new(false)
    }
}

#[pymodule]
fn webgame_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameWrapper>()?;
    m.add_class::<ObservableObject>()?;
    m.add_class::<GameState>()?;
    m.add_class::<AgentState>()?;
    m.add_class::<PyVec2>()?;
    Ok(())
}

use bevy::{log::LogPlugin, prelude::*};
use pyo3::prelude::*;
use webgame_game::{
    cartpole::{CartpoleState, NextAction},
    configs::LibCfgPlugin,
};

const THETA_THRESHOLD_RADIANS: f32 = 12.0 * 2.0 * std::f32::consts::PI / 360.0;
const X_THRESHOLD: f32 = 2.4;

type State = (f32, f32, f32, f32);

/// Rust implementation of the cartpole environment.
/// Simulates the Cartpole problem with Bevy.
#[pyclass]
pub struct CartpoleEnv {
    pub app: App,
}

#[pymethods]
impl CartpoleEnv {
    #[new]
    pub fn new() -> CartpoleEnv {
        let mut app = App::new();
        app.add_plugins(LibCfgPlugin);

        app.finish();
        app.cleanup();
        app.update();

        CartpoleEnv { app }
    }

    pub fn step(&mut self, action: u32) -> (State, f32, bool) {
        // Set next action, then get game state
        let mut next_action = self.app.world.get_resource_mut::<NextAction>().unwrap();
        next_action.0 = action;
        self.app.update();
        let game_state = self.get_state();

        let terminated = !(-X_THRESHOLD..=X_THRESHOLD).contains(&game_state.cart_pos)
            || !(-THETA_THRESHOLD_RADIANS..=THETA_THRESHOLD_RADIANS)
                .contains(&game_state.pole_angle);
        let reward = 1.0;

        (
            (
                game_state.cart_pos,
                game_state.cart_vel,
                game_state.pole_angle,
                game_state.pole_angvel,
            ),
            reward,
            terminated,
        )
    }

    pub fn reset(&mut self) -> State {
        *self = CartpoleEnv::new();
        let game_state = self.get_state();
        (
            game_state.cart_pos,
            game_state.cart_vel,
            game_state.pole_angle,
            game_state.pole_angvel,
        )
    }
}

impl CartpoleEnv {
    /// Returns the current game state.
    fn get_state(&self) -> &CartpoleState {
        self.app.world.get_resource::<CartpoleState>().unwrap()
    }
}

impl Default for CartpoleEnv {
    fn default() -> Self {
        Self::new()
    }
}

#[pymodule]
fn webgame_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CartpoleEnv>()?;
    Ok(())
}

//! Defines various configurations our game can be in.

use std::time::Duration;

use bevy::{input::InputPlugin, prelude::*, scene::ScenePlugin, time::TimeUpdateStrategy};

use crate::cartpole::{CartpolePlayPlugin, CartpolePlugin};

/// Handles core functionality for our game (i.e. gameplay logic).
pub struct CoreGamePlugin;

impl Plugin for CoreGamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CartpolePlugin);
    }
}

/// Adds functionality required to make the game playable (e.g. graphics and input handling).
pub struct PlayablePlugin;

impl Plugin for PlayablePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Your Project (Game)".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(CartpolePlayPlugin);
    }
}

/// The configuration for published builds.
pub struct ReleaseCfgPlugin;

impl Plugin for ReleaseCfgPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((PlayablePlugin, CoreGamePlugin));
    }
}

/// The configuration for library builds (e.g. for machine learning).
pub struct LibCfgPlugin;

impl Plugin for LibCfgPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            HierarchyPlugin,
            InputPlugin,
            AssetPlugin::default(),
            ScenePlugin,
            CoreGamePlugin,
        ))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            0.02,
        ))); // Use constant timestep
    }
}

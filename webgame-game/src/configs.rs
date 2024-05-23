//! Defines various configurations our game can be in.

use std::time::Duration;

use bevy::{
    asset::AssetMetaCheck,
    prelude::*,
    render::{settings::WgpuSettings, RenderPlugin},
    time::TimeUpdateStrategy,
    winit::WinitPlugin,
};
use bevy_rapier2d::prelude::*;

use crate::{
    gridworld::{GridworldPlayPlugin, GridworldPlugin},
    net::NetPlugin,
    observer::{ObserverPlayPlugin, ObserverPlugin},
    world_objs::WorldObjPlugin,
};

/// Handles core functionality for our game (i.e. gameplay logic).
pub struct CoreGamePlugin;

impl Plugin for CoreGamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
            .add_plugins((NetPlugin, GridworldPlugin, ObserverPlugin, WorldObjPlugin));
    }
}

/// Adds functionality required to make the game playable (e.g. graphics and input handling).
pub struct PlayablePlugin;

impl Plugin for PlayablePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AssetMetaCheck::Never)
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Your Project (Game)".into(),
                    resolution: (640., 360.).into(),
                    ..default()
                }),
                ..default()
            }))
            .add_plugins(RapierDebugRenderPlugin::default())
            .add_plugins((GridworldPlayPlugin, ObserverPlayPlugin));
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

const FIXED_TS: f32 = 0.02;

impl Plugin for LibCfgPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    close_when_requested: false,
                })
                .set(ImagePlugin::default_nearest())
                .set(RenderPlugin {
                    render_creation: WgpuSettings {
                        backends: None,
                        ..default()
                    }
                    .into(),
                    ..default()
                })
                .disable::<WinitPlugin>(),
            CoreGamePlugin,
        ))
        // Use constant timestep
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            FIXED_TS,
        )))
        .insert_resource(RapierConfiguration {
            timestep_mode: TimestepMode::Fixed {
                dt: FIXED_TS,
                substeps: 10,
            },
            ..default()
        });
    }
}

/// Optional plugin for library builds, adds support for Rerun visuals.
pub struct VisualizerPlugin;

impl Plugin for VisualizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins({
            let rec = revy::RecordingStreamBuilder::new("Pursuer")
                .spawn()
                .unwrap();
            revy::RerunPlugin { rec }
        });
    }
}

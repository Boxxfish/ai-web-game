//! Defines various configurations our game can be in.

use std::time::Duration;

use bevy::{
    asset::AssetMetaCheck, core::TaskPoolThreadAssignmentPolicy, prelude::*, render::{settings::WgpuSettings, RenderPlugin}, tasks::available_parallelism, time::TimeUpdateStrategy, winit::WinitPlugin
};
use bevy_rapier2d::prelude::*;

use crate::{
    agents::{AgentPlayPlugin, AgentPlugin},
    filter::{FilterPlayPlugin, FilterPlugin},
    gridworld::{GridworldPlayPlugin, GridworldPlugin, LevelLoader},
    net::NetPlugin,
    observer::{ObserverPlayPlugin, ObserverPlugin},
    screens::{ScreenState, ScreensPlayPlugin},
    ui::UIPlugin,
    world_objs::{WorldObjPlayPlugin, WorldObjPlugin},
};

/// Handles core functionality for our game (i.e. gameplay logic).
pub struct CoreGamePlugin;

impl Plugin for CoreGamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
            .add_plugins((
                NetPlugin,
                GridworldPlugin,
                ObserverPlugin,
                WorldObjPlugin,
                FilterPlugin,
                AgentPlugin,
            ))
            .insert_resource(RapierConfiguration {
                gravity: Vec2::ZERO,
                ..default()
            });
    }
}

/// Adds functionality required to make the game playable (e.g. graphics and input handling).
pub struct PlayablePlugin;

/// Marker resource that indicates we are playing a playable version of the game.
#[derive(Resource)]
pub struct IsPlayable;

impl Plugin for PlayablePlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(ScreenState::Loading)
            .insert_resource(IsPlayable)
            .insert_resource(AssetMetaCheck::Never)
            .insert_resource(ClearColor(Color::BLACK))
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Demo: Pursuer".into(),
                    resolution: (640., 360.).into(),
                    ..default()
                }),
                ..default()
            }))
            .add_plugins((
                WorldObjPlayPlugin,
                GridworldPlayPlugin,
                ObserverPlayPlugin,
                FilterPlayPlugin,
                AgentPlayPlugin,
                ScreensPlayPlugin,
                UIPlugin,
            ));
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

const FIXED_TS: f32 = 0.4;

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
                .set(TaskPoolPlugin {
                    task_pool_options: TaskPoolOptions {
                        min_total_threads: 1,
                        max_total_threads: std::usize::MAX,
                        io: TaskPoolThreadAssignmentPolicy {
                            min_threads: 1,
                            max_threads: 1,
                            percent: 0.,
                        },
                        async_compute: TaskPoolThreadAssignmentPolicy {
                            min_threads: 1,
                            max_threads: 1,
                            percent: 0.,
                        },
                        compute: TaskPoolThreadAssignmentPolicy {
                            min_threads: 1,
                            max_threads: 4,
                            percent: 1.0,
                        },
                    },
                })
                .disable::<WinitPlugin>(),
            CoreGamePlugin,
        ))
        // Use constant timestep
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            FIXED_TS,
        )))
        .insert_resource(RapierConfiguration {
            gravity: Vec2::ZERO,
            timestep_mode: TimestepMode::Fixed {
                dt: FIXED_TS,
                substeps: 10,
            },
            ..default()
        })
        .insert_state(ScreenState::Game);
    }
}

/// Optional plugin for library builds, adds support for Rerun visuals.
#[cfg(feature = "revy")]
pub struct VisualizerPlugin {
    pub recording_id: Option<String>,
}

#[cfg(feature = "revy")]
impl Plugin for VisualizerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins({
            let mut rec = revy::RecordingStreamBuilder::new("Pursuer");
            if let Some(recording_id) = &self.recording_id {
                rec = rec.recording_id(recording_id);
            }
            let rec = rec.spawn().unwrap();
            revy::RerunPlugin { rec }
        });
    }
}

use crate::{
    agents::{Agent, NextAction, PlayerAgent},
    gridworld::GRID_CELL_SIZE,
    observer::Wall,
};
use bevy::{prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::prelude::*;

/// Plugin for world objects (e.g. doors, noise sources).
pub struct WorldObjPlugin;

impl Plugin for WorldObjPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                handle_key_touch,
                update_noise_src,
                // visualize_noise_src,
                // visualize_visual_marker,
            ),
        );
    }
}

pub struct WorldObjPlayPlugin;

impl Plugin for WorldObjPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            key_idle_anim,
        );
    }
}

/// A key for unlocking doors.
#[derive(Component)]
pub struct Key;

/// Adds an idle animation to keys.
pub fn key_idle_anim(mut key_query: Query<&mut Transform, With<Key>>, time: Res<Time>) {
    for mut key_xform in key_query.iter_mut() {
        key_xform.translation.z = time.elapsed_seconds_wrapped().cos() * 10. + 5.;
    }
}

/// A door that can be opened and closed.
#[derive(Component, Default)]
pub struct Door {
    pub open: bool,
}

/// A visual indicating the door iteself.
#[derive(Component)]
pub struct DoorVisual;

/// How close the agent needs to be before it can pick up a key.
const PICKUP_DIST: f32 = GRID_CELL_SIZE / 2.;

/// Opens the door and destroys the key if the player touches it.
fn handle_key_touch(
    player_query: Query<&GlobalTransform, With<PlayerAgent>>,
    key_query: Query<(Entity, &GlobalTransform), With<Key>>,
    mut door_query: Query<(Entity, &mut Door)>,
    door_vis_query: Query<Entity, With<DoorVisual>>,
    mut commands: Commands,
) {
    for player_xform in player_query.iter() {
        let player_pos = player_xform.translation().xy();
        for (key_e, key_xform) in key_query.iter() {
            let obj_pos = key_xform.translation().xy();
            let dist_sq = (obj_pos - player_pos).length_squared();
            if dist_sq < PICKUP_DIST.powi(2) {
                commands.entity(key_e).despawn_recursive();
                for (door_e, mut door) in door_query.iter_mut() {
                    door.open = !door.open;
                    commands.entity(door_e).remove::<(Wall, Collider)>();
                }
                for vis_e in door_vis_query.iter() {
                    commands.entity(vis_e).despawn_recursive();
                }
            }
        }
    }
}

/// A source of noise that alerts observers within a radius.
#[derive(Component)]
pub struct NoiseSource {
    /// How far away to broadcast the noise.
    pub noise_radius: f32,
    /// How close an agent has to be to activate the noise source.
    pub active_radius: f32,
    pub activated_by: Option<Entity>,
}

/// Broadcasts that an agent touched the noise source.
fn update_noise_src(
    agent_query: Query<(Entity, &GlobalTransform), With<Agent>>,
    mut noise_query: Query<(&GlobalTransform, &mut NoiseSource)>,
) {
    for (obj_xform, mut noise) in noise_query.iter_mut() {
        noise.activated_by = None;
        for (agent_e, agent_xform) in agent_query.iter() {
            let agent_pos = agent_xform.translation().xy();
            let obj_pos = obj_xform.translation().xy();
            let dist_sq = (obj_pos - agent_pos).length_squared();
            if dist_sq <= noise.active_radius.powi(2) {
                noise.activated_by = Some(agent_e);
            }
        }
    }
}

/// Visualizes a noise source.
#[allow(dead_code)]
fn visualize_noise_src(mut gizmos: Gizmos, noise_query: Query<(&GlobalTransform, &NoiseSource)>) {
    for (obj_xform, noise) in noise_query.iter() {
        let obj_pos = obj_xform.translation().xy();
        gizmos.circle(
            obj_pos.extend(GRID_CELL_SIZE),
            Direction3d::Z,
            noise.active_radius,
            Color::BLUE,
        );
        if noise.activated_by.is_some() {
            gizmos.circle(
                obj_pos.extend(GRID_CELL_SIZE),
                Direction3d::Z,
                noise.noise_radius,
                Color::ORANGE,
            );
        }
    }
}

/// A visual marker.
/// Observers record the last seen positions of these items.
#[derive(Component)]
pub struct VisualMarker;

/// Visualizes a visual marker.
#[allow(dead_code)]
fn visualize_visual_marker(
    mut gizmos: Gizmos,
    visual_query: Query<&GlobalTransform, With<VisualMarker>>,
) {
    for obj_xform in visual_query.iter() {
        let obj_pos = obj_xform.translation().xy();
        gizmos.rect(
            obj_pos.extend(GRID_CELL_SIZE),
            Quat::IDENTITY,
            Vec2::ONE * GRID_CELL_SIZE * 0.5,
            Color::RED,
        );
    }
}

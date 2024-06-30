use crate::{
    gridworld::{Agent, NextAction, GRID_CELL_SIZE},
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
                update_door,
                visualize_door,
                update_noise_src,
                // visualize_noise_src,
                // visualize_visual_marker,
            ),
        );
    }
}

/// A door that can be opened and closed.
#[derive(Component, Default)]
pub struct Door {
    pub open: bool,
}

/// How close an object needs to be before the agent can toggle it.
const TOGGLE_DIST: f32 = GRID_CELL_SIZE * 1.5;

/// Opens and closes the door if the agent is not touching the door and it toggles nearby objects.
fn update_door(
    mut commands: Commands,
    agent_query: Query<(&GlobalTransform, &NextAction)>,
    mut door_query: Query<(Entity, &GlobalTransform, &mut Door)>,
) {
    for (agent_xform, action) in agent_query.iter() {
        let agent_pos = agent_xform.translation().xy();
        if action.toggle_objs {
            for (e, obj_xform, mut door) in door_query.iter_mut() {
                let obj_pos = obj_xform.translation().xy();
                let dist_sq = (obj_pos - agent_pos).length_squared();
                if dist_sq >= (GRID_CELL_SIZE / 2.).powi(2) && dist_sq < TOGGLE_DIST.powi(2) {
                    door.open = !door.open;
                    if door.open {
                        commands.entity(e).remove::<(Wall, Collider)>();
                    } else {
                        commands.entity(e).insert((
                            Wall,
                            Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                        ));
                    }
                }
            }
        }
    }
}

/// Updates the door visual.
fn visualize_door(
    mut commands: Commands,
    mut door_query: Query<(Entity, &Door, Option<&Handle<ColorMaterial>>), Changed<Door>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (e, door, mat_handle) in door_query.iter_mut() {
        if door.open {
            materials
                .get_mut(mat_handle.unwrap())
                .unwrap()
                .color
                .set_a(0.5);
            commands.entity(e).remove::<Wall>();
        } else if let Some(mat_handle) = mat_handle {
            materials.get_mut(mat_handle).unwrap().color.set_a(1.0);
            commands.entity(e).insert(Wall);
        } else {
            commands.entity(e).insert((
                Mesh2dHandle(meshes.add(Rectangle::new(GRID_CELL_SIZE, GRID_CELL_SIZE))),
                materials.add(Color::MAROON),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
            commands.entity(e).insert(Wall);
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
fn visualize_visual_marker(
    mut gizmos: Gizmos,
    visual_query: Query<(&GlobalTransform, &VisualMarker)>,
) {
    for (obj_xform, visual) in visual_query.iter() {
        let obj_pos = obj_xform.translation().xy();
        gizmos.rect(
            obj_pos.extend(GRID_CELL_SIZE),
            Quat::IDENTITY,
            Vec2::ONE * GRID_CELL_SIZE * 0.5,
            Color::RED,
        );
    }
}

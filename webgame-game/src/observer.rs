use std::collections::HashMap;

use bevy::{prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::{math::Real, prelude::*};
use ordered_float::OrderedFloat;

use crate::{gridworld::{move_agents, Agent}, world_objs::VisualMarker};

/// Plugins for determining what agents can see.
pub struct ObserverPlugin;

impl Plugin for ObserverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                update_observers.after(move_agents),
                update_vm_data,
                draw_observer_areas.after(update_observers),
            ),
        );
    }
}

/// Implements playable functionality for ObserverPlugin.
pub struct ObserverPlayPlugin;

impl Plugin for ObserverPlayPlugin {
    fn build(&self, _app: &mut App) {}
}

/// Stores visual marker data for an observer
pub struct VMSeenData {
    /// When it was last seen (time since startup).
    pub last_seen: f32,
    /// When it was last seen (elapsed, only exists when `last_state` is true)
    pub last_seen_elapsed: Option<f32>,
    /// The state when this was last seen.
    pub last_state: bool,
    /// Whether the state of this has changed since it was last seen.
    pub state_changed: bool,
}

/// Indicates that this entity can observe observable entities.
#[derive(Default, Component)]
pub struct Observer {
    /// Entities the observer can see.
    pub observing: Vec<Entity>,
    /// Stores data on visual markers that it's seen.
    pub seen_markers: HashMap<Entity, VMSeenData>,
    /// Stores a list of triangles that make up the observer's field of vision.
    pub vis_mesh: Vec<[Vec2; 3]>,
}

/// Indicates that this entity can be observed.
#[derive(Component)]
pub struct Observable;

/// Causes debug info for this observer to be displayed.
#[derive(Component)]
pub struct DebugObserver;

/// Blocks the observer's field of view.
/// Currently, only supports entities with rect colliders.
#[derive(Component)]
pub struct Wall;

/// Updates observers with observable entities they can see.
fn update_observers(
    wall_query: Query<(Entity, &Transform, &Collider), With<Wall>>,
    mut observer_query: Query<(Entity, &mut Observer, &Transform, &Agent)>,
    observable_query: Query<(Entity, &Transform), With<Observable>>,
    rapier_ctx: Res<RapierContext>,
) {
    // Collect wall endpoints
    let mut all_endpoints = Vec::new();
    for (_, wall_xform, wall_c) in wall_query.iter() {
        let rect = wall_c.as_cuboid().unwrap();
        let half = rect.raw.half_extents.xy();
        let x_axis = wall_xform.right().xy();
        let y_axis = wall_xform.up().xy();
        let center = wall_xform.translation.xy();
        let endpoints = (0..4)
            .map(|i| (((i % 2) * 2 - 1) as f32, ((i / 2) * 2 - 1) as f32))
            .map(|(x_sign, y_sign)| center + x_sign * x_axis * half.x + y_sign * y_axis * half.y)
            .collect::<Vec<_>>();
        all_endpoints.extend_from_slice(&endpoints);
    }

    // Draw per agent visibility triangles
    let walls = wall_query.iter().map(|(e, _, _)| e).collect::<Vec<_>>();
    for (observer_e, mut observer, observer_xform, agent) in observer_query.iter_mut() {
        // Draw vision cone
        let fov = 60_f32.to_radians();
        let start = observer_xform.translation.xy();
        let cone_l = Mat2::from_angle(-fov / 2.) * agent.dir;
        let cone_r = Mat2::from_angle(fov / 2.) * agent.dir;

        // Add cone boundaries to endpoints
        let mut sorted_endpoints = all_endpoints.clone();
        sorted_endpoints.extend_from_slice(&[start + cone_l, start + cone_r]);

        // Sort endpoints by angle and remove any points not within the vision cone
        sorted_endpoints.retain_mut(|p| {
            let dir = (*p - start).normalize();
            dir.dot(agent.dir).acos() <= fov / 2. + 0.01
        });
        sorted_endpoints.sort_unstable_by_key(|p| {
            let dir = (*p - start).normalize();
            OrderedFloat(dir.x * -dir.y.signum() - dir.y.signum())
        });

        let first_idx = sorted_endpoints
            .iter()
            .position(|p| p.abs_diff_eq(start + cone_l, 0.1))
            .unwrap_or(0);

        // Sweep from `cone_l` to `cone_r`
        let mut all_tris = Vec::new();
        for i in 0..sorted_endpoints.len() {
            let i = (i + first_idx) % sorted_endpoints.len();
            let p = sorted_endpoints[i];
            let dir = (p - start).normalize();
            let mut tri = Vec::new();
            for mat in [Mat2::from_angle(-0.001), Mat2::from_angle(0.001)] {
                let dir = mat * dir;
                let result = rapier_ctx.cast_ray(
                    start,
                    dir,
                    Real::MAX,
                    false,
                    QueryFilter::new().predicate(&|e| walls.contains(&e)),
                );
                if let Some((_, dist)) = result {
                    tri.push(start + dir * dist);
                }
            }
            if tri.len() == 2 {
                all_tris.push(tri);
            }
        }

        // Generate new vision mesh
        let mut vis_mesh = Vec::new();
        if !all_tris.is_empty() {
            for i in 0..(all_tris.len() - 1) {
                let next_i = (i + 1) % all_tris.len();
                let tri = &all_tris[i];
                let next_tri = &all_tris[next_i];
                vis_mesh.push([start, tri[1], next_tri[0]]);
            }
        }
        observer.vis_mesh = vis_mesh;

        // Check which observable objects fall within the mesh
        let mut observing = Vec::new();
        for (observable_e, observable_xform) in observable_query.iter() {
            if observable_e == observer_e {
                continue;
            }

            let p = observable_xform.translation.xy();
            for tri in &observer.vis_mesh {
                let d1 = sign(p, tri[0], tri[1]);
                let d2 = sign(p, tri[1], tri[2]);
                let d3 = sign(p, tri[2], tri[0]);

                let has_neg = d1 < 0. || d2 < 0. || d3 < 0.;
                let has_pos = d1 > 0. || d2 > 0. || d3 > 0.;

                if !(has_neg && has_pos) {
                    observing.push(observable_e);
                    break;
                }
            }
        }
        observer.observing = observing;
    }
}

/// Updates observers' visual marker data.
fn update_vm_data(
    mut observer_query: Query<&mut Observer>,
    visual_query: Query<(Entity, &VisualMarker)>,
    time: Res<Time>,
) {
    for mut observer in observer_query.iter_mut() {
        for (v_e, v_marker) in visual_query.iter() {
            if observer.observing.contains(&v_e) {
                if let Some(vm_data) = observer.seen_markers.get_mut(&v_e) {
                    vm_data.last_seen_elapsed =
                        Some(time.elapsed_seconds_wrapped() - vm_data.last_seen);
                    vm_data.last_seen = time.elapsed_seconds_wrapped();
                    vm_data.state_changed = vm_data.last_state != v_marker.state;
                    vm_data.last_state = v_marker.state;
                } else {
                    observer.seen_markers.insert(
                        v_e,
                        VMSeenData {
                            last_seen: time.elapsed_seconds_wrapped(),
                            last_state: v_marker.state,
                            state_changed: false,
                            last_seen_elapsed: Some(0.),
                        },
                    );
                }
            } else if let Some(vm_data) = observer.seen_markers.get_mut(&v_e) {
                vm_data.state_changed = false;
                vm_data.last_seen_elapsed = None;
            } else {
                observer.seen_markers.insert(
                    v_e,
                    VMSeenData {
                        last_seen: time.elapsed_seconds_wrapped(),
                        last_state: true,
                        state_changed: false,
                        last_seen_elapsed: None,
                    },
                );
            }
        }
    }
}

/// Helper function for detecting if a point is in a triangle.
fn sign(p1: Vec2, p2: Vec2, p3: Vec2) -> f32 {
    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
}

/// Marker component for vision cone visual.
#[derive(Component)]
struct VisCone;

/// Draws visible areas for observers.
fn draw_observer_areas(
    observer_query: Query<&Observer, With<DebugObserver>>,
    vis_cone_query: Query<(Entity, &Mesh2dHandle, &Handle<ColorMaterial>), With<VisCone>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (e, mesh, color) in vis_cone_query.iter() {
        meshes.remove(&mesh.0);
        materials.remove(color);
        commands.entity(e).despawn();
    }
    for observer in observer_query.iter() {
        for tri in &observer.vis_mesh {
            commands.spawn((
                ColorMesh2dBundle {
                    mesh: Mesh2dHandle(meshes.add(Triangle2d::new(tri[0], tri[1], tri[2]))),
                    material: materials.add(Color::YELLOW.with_a(0.01)),
                    ..default()
                },
                VisCone,
            ));
        }
    }
}

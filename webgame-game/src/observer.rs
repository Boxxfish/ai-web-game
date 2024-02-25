use bevy::prelude::*;
use bevy_rapier2d::{math::Real, na::ComplexField, prelude::*};
use ordered_float::OrderedFloat;

use crate::gridworld::Agent;

/// Plugins for determining what agents can see.
pub struct ObserverPlugin;

impl Plugin for ObserverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_observers);
    }
}

/// Implements playable functionality for ObserverPlugin.
pub struct ObserverPlayPlugin;

impl Plugin for ObserverPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_observer_areas);
    }
}

/// Indicates that this entity can observe observable entities.
#[derive(Default, Component)]
pub struct Observer {
    pub observing: Vec<Entity>,
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
fn update_observers() {}

/// Draws parts of the room observers can see.
#[allow(clippy::type_complexity)]
fn draw_observer_areas(
    wall_query: Query<(Entity, &Transform, &Collider), With<Wall>>,
    observer_query: Query<(Entity, &Transform, &Agent), (With<Observer>, With<DebugObserver>)>,
    rapier_ctx: Res<RapierContext>,
    mut gizmos: Gizmos,
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
    for (_, observer_xform, agent) in observer_query.iter() {
        // Draw vision cone
        let fov = 60_f32.to_radians();
        let start = observer_xform.translation.xy();
        let cone_l = Mat2::from_angle(-fov / 2.) * agent.dir;
        let cone_r = Mat2::from_angle(fov / 2.) * agent.dir;
        gizmos.line_2d(start, start + agent.dir * 10., Color::AQUAMARINE);
        gizmos.line_2d(start, start + cone_l * 40., Color::YELLOW);
        gizmos.line_2d(start, start + cone_r * 40., Color::YELLOW);

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

        let mut all_tris = Vec::new();
        for p in &sorted_endpoints {
            let dir = (*p - start).normalize();
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
        if !all_tris.is_empty() {
            for i in 0..all_tris.len() {
                let next_i = (i + 1) % all_tris.len();
                let tri = &all_tris[i];
                let next_tri = &all_tris[next_i];
                gizmos.linestrip_2d(
                    [start, tri[1], next_tri[0], start],
                    Color::YELLOW.with_a(0.01),
                );
            }
        }
    }
}
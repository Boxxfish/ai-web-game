use bevy::prelude::*;
use bevy_rapier2d::{math::Real, prelude::*};
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
    mut observer_query: Query<(&mut Observer, &Transform, &Agent)>,
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
    for (mut observer, observer_xform, agent) in observer_query.iter_mut() {
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
    }
}

/// Draws parts of the room observers can see.
#[allow(clippy::type_complexity)]
fn draw_observer_areas(
    observer_query: Query<(&Observer, &Transform, &Agent), With<DebugObserver>>,
    mut gizmos: Gizmos,
) {
    for (observer, observer_xform, agent) in observer_query.iter() {
        // Draw vision cone
        let fov = 60_f32.to_radians();
        let start = observer_xform.translation.xy();
        let cone_l = Mat2::from_angle(-fov / 2.) * agent.dir;
        let cone_r = Mat2::from_angle(fov / 2.) * agent.dir;
        gizmos.line_2d(start, start + agent.dir * 10., Color::AQUAMARINE);
        gizmos.line_2d(start, start + cone_l * 40., Color::YELLOW);
        gizmos.line_2d(start, start + cone_r * 40., Color::YELLOW);

        // Draw visible areas
        for tri in &observer.vis_mesh {
            gizmos.linestrip_2d([tri[0], tri[1], tri[2], tri[0]], Color::YELLOW.with_a(0.01));
        }
    }
}

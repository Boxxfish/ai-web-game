use bevy::prelude::*;
use bevy_rapier2d::{math::Real, prelude::*};
use ordered_float::OrderedFloat;

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
        app.add_systems(Update, (draw_observer_areas, draw_observed));
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
fn draw_observer_areas(
    wall_query: Query<(&Transform, &Collider), With<Wall>>,
    observer_query: Query<(Entity, &Transform), With<Observer>>,
    rapier_ctx: Res<RapierContext>,
    mut gizmos: Gizmos,
) {
    // Draw wall endpoints
    let point_size = 1.;
    let mut all_endpoints = Vec::new();
    let mut all_walls = Vec::new();
    for (wall_xform, wall_c) in wall_query.iter() {
        let rect = wall_c.as_cuboid().unwrap();
        let half = rect.raw.half_extents.xy();
        let x_axis = wall_xform.right().xy();
        let y_axis = wall_xform.up().xy();
        let center = wall_xform.translation.xy();
        let endpoints = (0..4)
            .map(|i| (((i % 2) * 2 - 1) as f32, ((i / 2) * 2 - 1) as f32))
            .map(|(x_sign, y_sign)| center + x_sign * x_axis * half.x + y_sign * y_axis * half.y)
            .collect::<Vec<_>>();
        for p in &endpoints {
            gizmos.circle_2d(*p, point_size, Color::GREEN);
        }
        all_endpoints.extend_from_slice(&endpoints);
        all_walls.extend_from_slice(&[
            [endpoints[0], endpoints[1]],
            [endpoints[1], endpoints[2]],
            [endpoints[2], endpoints[3]],
            [endpoints[3], endpoints[0]],
        ]);
    }

    // Draw per agent visibility triangles
    for (observer_e, observer_xform) in observer_query.iter() {
        // Sort endpoints by angle
        let start = observer_xform.translation.xy();
        let mut sorted_endpoints = all_endpoints.clone();
        sorted_endpoints.sort_unstable_by_key(|p| {
            let dir = (*p - start).normalize();
            OrderedFloat(dir.x + 2. * dir.y.signum())
        });

        let mut dists = Vec::new();
        for p in &sorted_endpoints {
            let dir = (*p - start).normalize();
            let result = rapier_ctx.cast_ray(
                start,
                dir,
                Real::MAX,
                false,
                QueryFilter::new().exclude_collider(observer_e),
            );
            if let Some((_, dist)) = result {
                dists.push(dist);
            }
        }
        for i in 0..dists.len() {
            let next_i = (i + 1) % dists.len();
            let dist = dists[i];
            let next_dist = dists[next_i];
            let dir = (sorted_endpoints[i] - start).normalize();
            let next_dir = (sorted_endpoints[next_i] - start).normalize();
            gizmos.linestrip_2d(
                [
                    start,
                    start + next_dir * next_dist,
                    start + dir * dist,
                    start,
                ],
                Color::YELLOW.with_a(0.2),
            );
        }
    }
}

/// Indicates which observers are looking at which observable entities.
fn draw_observed() {}

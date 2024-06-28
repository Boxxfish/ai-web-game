use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_rapier2d::{
    control::KinematicCharacterController,
    dynamics::{Damping, LockedAxes, RigidBody},
    geometry::Collider,
};
use rand::{seq::IteratorRandom, Rng};

use crate::{
    configs::IsPlayable,
    observer::{DebugObserver, Observable, Observer, Wall},
    world_objs::{NoiseSource, VisualMarker},
};

/// Plugin for basic game features, such as moving around and not going through walls.
pub struct GridworldPlugin;

impl Plugin for GridworldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_entities).add_systems(
            Update,
            (
                move_agents,
                visualize_agent::<PursuerAgent>(Color::RED),
                visualize_agent::<PlayerAgent>(Color::GREEN),
            ),
        );
    }
}

/// Adds playable functionality for `GridworldPlugin`.
pub struct GridworldPlayPlugin;

impl Plugin for GridworldPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, set_player_action);
    }
}

/// The width and height of the level by default.
pub const DEFAULT_LEVEL_SIZE: usize = 8;
/// The probability of a door spawning in an empty cell.
pub const DOOR_PROB: f64 = 0.05;

/// Stores the layout of the level.
#[derive(Resource)]
pub struct LevelLayout {
    /// Stores `true` if a wall exists, `false` for empty spaces. The first element is the top right corner.
    pub walls: Vec<bool>,
    pub size: usize,
}

impl LevelLayout {
    /// Generates a randomized level.
    pub fn random(size: usize, wall_prob: f64) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            walls: (0..(size * size))
                .map(|_| rng.gen_bool(wall_prob))
                .collect(),
            size,
        }
    }

    /// Returns a random empty tile index.
    pub fn get_empty(&self) -> usize {
        let mut rng = rand::thread_rng();
        let tile_idx = self
            .walls
            .iter()
            .enumerate()
            .filter(|(_, x)| !**x)
            .map(|(x, _)| x)
            .choose(&mut rng)
            .unwrap();
        tile_idx
    }
}

/// State used by all agents.
#[derive(Component, Clone, Copy)]
pub struct Agent {
    /// The direction the agent is currently looking at.
    pub dir: Vec2,
}

impl Default for Agent {
    fn default() -> Self {
        Self { dir: Vec2::X }
    }
}

// Indicates the Pursuer agent.
#[derive(Component)]
pub struct PursuerAgent;

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// Determines the maximum number of items that will be spawned.
/// If this resource is not found, no items are spawned.
#[derive(Resource)]
pub struct MaxItems(pub usize);

/// Sets up all entities in the game.
fn setup_entities(
    mut commands: Commands,
    level: Res<LevelLayout>,
    max_items: Option<Res<MaxItems>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    is_playable: Option<Res<IsPlayable>>,
) {
    // Add camera + light
    commands.spawn(Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            700.,
        ))
        .looking_to(-Vec3::Z, Vec3::Y),
        projection: Projection::Perspective(PerspectiveProjection {
            fov: 0.4,
            ..default()
        }),
        ..default()
    });
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 2000.,
            ..default()
        },
        ..default()
    });

    let pursuer_tile_idx = level.get_empty();
    commands
        .spawn((
            PursuerAgent,
            Agent::default(),
            NextAction::default(),
            Collider::ball(GRID_CELL_SIZE * 0.25),
            RigidBody::KinematicPositionBased,
            KinematicCharacterController::default(),
            TransformBundle::from_transform(Transform::from_translation(
                Vec3::new(
                    (pursuer_tile_idx % level.size) as f32,
                    (pursuer_tile_idx / level.size) as f32,
                    0.,
                ) * GRID_CELL_SIZE,
            )),
            Observer::default(),
            Observable,
            DebugObserver,
        ))
        .with_children(|p| {
            if is_playable.is_some() {
                p.spawn(SceneBundle {
                    scene: asset_server.load("characters/cyborgFemaleA.glb#Scene0"),
                    transform: Transform::default()
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                        .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.5),
                    ..default()
                });
            }
        });
    let player_tile_idx = level.get_empty();
    commands
        .spawn((
            PlayerAgent,
            Agent::default(),
            NextAction::default(),
            Collider::ball(GRID_CELL_SIZE * 0.25),
            RigidBody::KinematicPositionBased,
            KinematicCharacterController::default(),
            TransformBundle::from_transform(Transform::from_translation(
                Vec3::new(
                    (player_tile_idx % level.size) as f32,
                    (player_tile_idx / level.size) as f32,
                    0.,
                ) * GRID_CELL_SIZE,
            )),
            Observer::default(),
            Observable,
            DebugObserver,
        ))
        .with_children(|p| {
            if is_playable.is_some() {
                p.spawn(SceneBundle {
                    scene: asset_server.load("characters/skaterMaleA.glb#Scene0"),
                    transform: Transform::default()
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                        .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.5),
                    ..default()
                });
            }
        });

    // Add floor
    commands.spawn(SceneBundle {
        scene: asset_server.load("furniture/floorFull.glb#Scene0"),
        transform: Transform::default()
            .with_translation(Vec3::new(-1., -1., 0.) * GRID_CELL_SIZE / 2.)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
            .with_scale(Vec3::new(level.size as f32, 1., level.size as f32) * GRID_CELL_SIZE),
        ..default()
    });

    // Set up walls and doors
    let wall_mesh = meshes.add(Cuboid::new(GRID_CELL_SIZE, GRID_CELL_SIZE, GRID_CELL_SIZE));
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        unlit: true,
        ..default()
    });
    let mut rng = rand::thread_rng();
    for y in 0..level.size {
        for x in 0..level.size {
            if level.walls[y * level.size + x] {
                commands
                    .spawn((
                        Wall,
                        Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                        TransformBundle::from_transform(Transform::from_translation(
                            Vec3::new(x as f32, y as f32, 0.) * GRID_CELL_SIZE,
                        )),
                        VisibilityBundle::default(),
                    ))
                    .with_children(|p| {
                        if is_playable.is_some() {
                            let offsets = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y];
                            let base_xform = Transform::default()
                                .with_translation(-Vec3::X * GRID_CELL_SIZE / 2.)
                                .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                                .with_scale(Vec3::ONE * GRID_CELL_SIZE);
                            for i in 0..4 {
                                let should_spawn = match i {
                                    3 => (y > 0) && !level.walls[(y - 1) * level.size + x],
                                    2 => {
                                        (y < level.size - 1)
                                            && !level.walls[(y + 1) * level.size + x]
                                    }
                                    1 => (x > 0) && !level.walls[y * level.size + (x - 1)],
                                    0 => {
                                        (x < level.size - 1)
                                            && !level.walls[y * level.size + (x + 1)]
                                    }
                                    _ => unreachable!(),
                                };
                                if should_spawn {
                                    let rot = if i >= 2 {
                                        Quat::IDENTITY
                                    } else {
                                        Quat::from_rotation_z(std::f32::consts::PI / 2.)
                                    };
                                    p.spawn(SceneBundle {
                                        scene: asset_server.load("furniture/wall.glb#Scene0"),
                                        transform: Transform::default()
                                            .with_rotation(rot)
                                            .with_translation(offsets[i] * (GRID_CELL_SIZE / 2. + 0.1))
                                            * base_xform,
                                        ..default()
                                    });
                                }
                            }
                        }
                        p.spawn(PbrBundle {
                            mesh: wall_mesh.clone(),
                            material: wall_mat.clone(),
                            transform: Transform::from_translation(Vec3::Z * GRID_CELL_SIZE / 2.),
                            ..default()
                        });
                    });
            } else if rng.gen_bool(DOOR_PROB) {
                // commands.spawn((
                //     Door::default(),
                //     Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                //     TransformBundle::from_transform(Transform::from_translation(
                //         Vec3::new(x as f32, y as f32, 0.) * GRID_CELL_SIZE,
                //     )),
                // ));
            }
        }
    }

    // Set up the sides of the game world
    let half_sizes = [GRID_CELL_SIZE / 2., GRID_CELL_SIZE * level.size as f32 / 2.];
    let wall_positions = [-GRID_CELL_SIZE, GRID_CELL_SIZE * level.size as f32];
    let wall_pos_offset = GRID_CELL_SIZE * (level.size - 1) as f32 / 2.;
    for i in 0..4 {
        let positions = [wall_positions[i % 2], wall_pos_offset];
        commands
            .spawn((
                Wall,
                Collider::cuboid(half_sizes[i / 2], half_sizes[1 - i / 2]),
                TransformBundle::from_transform(Transform::from_translation(Vec3::new(
                    positions[i / 2],
                    positions[1 - i / 2],
                    0.,
                ))),
                VisibilityBundle::default(),
            ))
            .with_children(|p| {
                if is_playable.is_some() {
                    let offsets = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y];
                    let base_xform = Transform::default()
                        .with_translation(-Vec3::X * GRID_CELL_SIZE * level.size as f32 / 2.)
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                        .with_scale(Vec3::new(level.size as f32, 1., 1.) * GRID_CELL_SIZE);
                    let rot = if i >= 2 {
                        Quat::IDENTITY
                    } else {
                        Quat::from_rotation_z(std::f32::consts::PI / 2.)
                    };
                    p.spawn(SceneBundle {
                        scene: asset_server.load("furniture/wall.glb#Scene0"),
                        transform: Transform::default()
                            .with_rotation(rot)
                            .with_translation(offsets[i] * GRID_CELL_SIZE / 2.)
                            * base_xform,
                        ..default()
                    });
                } else {
                    p.spawn(PbrBundle {
                        mesh: meshes.add(Cuboid::new(
                            half_sizes[i / 2] * 2.,
                            half_sizes[1 - i / 2] * 2.,
                            GRID_CELL_SIZE,
                        )),
                        material: wall_mat.clone(),
                        ..default()
                    });
                }
            });
    }

    // Add noise sources and visual markers
    if let Some(max_items) = max_items {
        for _ in 0..rng.gen_range(0..max_items.0) {
            let tile_idx = level.get_empty();
            let y = tile_idx / level.size;
            let x = tile_idx % level.size;
            let pos = Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE
                + Vec3::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5, 0.) * GRID_CELL_SIZE;
            commands.spawn((
                RigidBody::Dynamic,
                Damping {
                    linear_damping: 10.,
                    ..default()
                },
                LockedAxes::ROTATION_LOCKED,
                Collider::cuboid(GRID_CELL_SIZE * 0.4, GRID_CELL_SIZE * 0.4),
                NoiseSource {
                    noise_radius: GRID_CELL_SIZE * 3.,
                    active_radius: GRID_CELL_SIZE * 1.5,
                    activated_by: None,
                },
                VisualMarker,
                Observable,
                TransformBundle::from_transform(Transform::from_translation(pos)),
            ));
        }
    }
}

pub const GRID_CELL_SIZE: f32 = 25.;

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(
    Commands<'_, '_>,
    Query<'_, '_, Entity, Added<T>>,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
) {
    move |mut commands: Commands,
          agent_query: Query<Entity, Added<T>>,
          mut meshes: ResMut<Assets<Mesh>>,
          mut materials: ResMut<Assets<ColorMaterial>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Mesh2dHandle(meshes.add(Circle::new(GRID_CELL_SIZE * 0.25))),
                materials.add(color),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
        }
    }
}

const AGENT_SPEED: f32 = GRID_CELL_SIZE * 2.;

/// Holds the next action for an agent.
#[derive(Default, Component)]
pub struct NextAction {
    /// Which direction the agent will move in.
    pub dir: Vec2,
    /// Whether the agent should toggle nearby objects this frame.
    pub toggle_objs: bool,
}

/// Allows the player to set the Players next action.
fn set_player_action(
    inpt: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut NextAction, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::KeyW) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::KeyS) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::KeyA) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::KeyD) {
        dir.x += 1.;
    }
    let mut next_action = player_query.single_mut();
    next_action.dir = dir;
    next_action.toggle_objs = false;
    if inpt.just_pressed(KeyCode::KeyF) {
        next_action.toggle_objs = true;
    }
}

/// Moves agents around.
pub fn move_agents(
    mut agent_query: Query<(&mut Agent, &mut KinematicCharacterController, &NextAction)>,
    time: Res<Time>,
) {
    for (mut agent, mut controller, next_action) in agent_query.iter_mut() {
        let dir = next_action.dir;
        if dir.length_squared() > 0.1 {
            let dir = dir.normalize();
            agent.dir = dir;
            controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
        }
    }
}

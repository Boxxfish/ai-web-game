use std::{f32::consts::PI, time::Duration};

use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    sprite::Mesh2dHandle,
    utils::BoxedFuture,
};
use bevy_rapier2d::{
    control::KinematicCharacterController,
    dynamics::{Damping, LockedAxes, RigidBody},
    geometry::Collider,
};
use rand::{seq::IteratorRandom, Rng};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    configs::IsPlayable,
    observer::{DebugObserver, Observable, Observer, Wall},
    world_objs::{NoiseSource, VisualMarker},
};

/// Plugin for basic game features, such as moving around and not going through walls.
pub struct GridworldPlugin;

impl Plugin for GridworldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                setup_entities.run_if(resource_added::<LevelLayout>),
                (
                    move_agents,
                    visualize_agent::<PursuerAgent>(Color::RED),
                    visualize_agent::<PlayerAgent>(Color::GREEN),
                )
                    .run_if(resource_exists::<ShouldRun>),
            ),
        );
    }
}

/// Adds playable functionality for `GridworldPlugin`.
pub struct GridworldPlayPlugin;

impl Plugin for GridworldPlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<LoadedLevelData>()
            .init_asset_loader::<LoadedLevelDataLoader>()
            .add_systems(
                Update,
                (
                    load_level,
                    set_player_action.run_if(resource_exists::<ShouldRun>),
                ),
            );
    }
}

/// The width and height of the level by default.
pub const DEFAULT_LEVEL_SIZE: usize = 8;
/// The probability of a door spawning in an empty cell.
pub const DOOR_PROB: f64 = 0.05;

/// Data for objects in levels.
#[derive(Deserialize, Clone)]
pub struct LoadedObjData {
    pub name: String,
    pub pos: (usize, usize),
    #[serde(default)]
    pub dir: Option<String>,
    #[serde(default)]
    pub movable: bool,
}

/// Data for loaded levels.
#[derive(Deserialize, Asset, TypePath)]
pub struct LoadedLevelData {
    pub size: usize,
    pub walls: Vec<u8>,
    pub objects: Vec<LoadedObjData>,
}

/// Indicates that a level should be loaded.
#[derive(Resource)]
pub enum LevelLoader {
    Path(String),
    Asset(Handle<LoadedLevelData>),
}

#[derive(Default)]
struct LoadedLevelDataLoader;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LoadedLevelDataLoaderError {
    #[error("Could not load asset: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse JSON: {0}")]
    RonSpannedError(#[from] serde_json::error::Error),
}

impl AssetLoader for LoadedLevelDataLoader {
    type Asset = LoadedLevelData;
    type Settings = ();
    type Error = LoadedLevelDataLoaderError;
    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut buf = String::new();
            reader.read_to_string(&mut buf).await?;
            let data: LoadedLevelData = serde_json::from_str(&buf)?;
            Ok(data)
        })
    }

    fn extensions(&self) -> &[&str] {
        &[".json"]
    }
}

/// Loads levels.
fn load_level(
    level: Option<Res<LevelLoader>>,
    level_data: Res<Assets<LoadedLevelData>>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
) {
    if let Some(level) = level {
        match level.as_ref() {
            LevelLoader::Path(path) => {
                commands.insert_resource(LevelLoader::Asset(asset_server.load(path)));
            }
            LevelLoader::Asset(handle) => {
                if let Some(level) = level_data.get(handle.clone()) {
                    let mut walls = Vec::new();
                    for y in 0..level.size {
                        for x in 0..level.size {
                            walls.push(level.walls[(level.size - y - 1) * level.size + x] != 0);
                        }
                    }
                    commands.insert_resource(LevelLayout {
                        walls,
                        size: level.size,
                        objects: level.objects.clone(),
                    });
                    commands.remove_resource::<LevelLoader>();
                }
            }
        }
    }
}

/// Indicates that the game should begin running.
#[derive(Resource)]
pub struct ShouldRun;

/// Stores the layout of the level.
#[derive(Resource)]
pub struct LevelLayout {
    /// Stores `true` if a wall exists, `false` for empty spaces. The first element is the top right corner.
    pub walls: Vec<bool>,
    pub size: usize,
    pub objects: Vec<LoadedObjData>,
}

impl LevelLayout {
    /// Generates a randomized level.
    pub fn random(size: usize, wall_prob: f64, max_items: usize) -> Self {
        let mut rng = rand::thread_rng();
        let orig = Self {
            walls: (0..(size * size))
                .map(|_| rng.gen_bool(wall_prob))
                .collect(),
            size,
            objects: Vec::new(),
        };
        let mut objects = Vec::new();
        for _ in 0..rng.gen_range(0..max_items) {
            let tile_idx = orig.get_empty();
            let y = tile_idx / size;
            let x = tile_idx % size;
            objects.push(LoadedObjData {
                name: "".into(),
                pos: (x, y),
                dir: Some("left".into()),
                movable: true,
            });
        }
        Self {
            walls: orig.walls,
            size,
            objects,
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

/// The child of an `Agent` that contains its visuals.
#[derive(Component)]
pub struct AgentVisuals;

/// Sets up all entities in the game.
fn setup_entities(
    mut commands: Commands,
    level: Res<LevelLayout>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    is_playable: Option<Res<IsPlayable>>,
) {
    // Add camera + light
    commands.spawn(Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(
            GRID_CELL_SIZE * (((level.size + 1) / 2) as f32),
            -300.,
            700.,
        ))
        .with_rotation(Quat::from_rotation_x(0.5)),
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
        transform: Transform::from_rotation(Quat::from_rotation_x(PI / 4.)),
        ..default()
    });

    let pursuer_tile_idx = 0;// level.get_empty();
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
                p.spawn((
                    AgentVisuals,
                    SceneBundle {
                        scene: asset_server.load("characters/cyborgFemaleA.glb#Scene0"),
                        transform: Transform::default()
                            .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                            .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.4),
                        ..default()
                    },
                ));
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
                p.spawn((
                    AgentVisuals,
                    SceneBundle {
                        scene: asset_server.load("characters/skaterMaleA.glb#Scene0"),
                        transform: Transform::default()
                            .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                            .with_scale(Vec3::ONE * GRID_CELL_SIZE * 0.4),
                        ..default()
                    },
                ));
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
                            for (i, offset) in offsets.iter().enumerate() {
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
                                            .with_translation(
                                                *offset * (GRID_CELL_SIZE / 2. + 0.1),
                                            )
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

    // Add objects, which may include noise sources and visual markers
    let obj_mat = materials.add(StandardMaterial {
        base_color: Color::BLUE,
        unlit: true,
        ..default()
    });
    for obj in &level.objects {
        let (x, y) = obj.pos;
        let pos = Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE;
        let collider_size = GRID_CELL_SIZE * 0.8;
        let e = commands
            .spawn((
                Collider::cuboid(collider_size / 2., collider_size / 2.),
                TransformBundle::from_transform(Transform::from_translation(pos)),
                VisibilityBundle::default(),
            ))
            .id();
        if obj.movable {
            commands.entity(e).insert((
                RigidBody::Dynamic,
                Damping {
                    linear_damping: 10.,
                    ..default()
                },
                LockedAxes::ROTATION_LOCKED,
                NoiseSource {
                    noise_radius: GRID_CELL_SIZE * 3.,
                    active_radius: GRID_CELL_SIZE * 1.5,
                    activated_by: None,
                },
                VisualMarker,
                Observable,
            ));
        }
        commands.entity(e).with_children(|p| {
            if is_playable.is_some() {
                let base_xform = Transform::default()
                    .with_translation(-Vec3::X * GRID_CELL_SIZE / 2.)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::PI / 2.))
                    .with_scale(Vec3::ONE * GRID_CELL_SIZE * 2.);
                let rot = match obj.dir.clone().unwrap_or("left".into()).as_str() {
                    "left" => Quat::from_rotation_z(std::f32::consts::PI * 3. / 2.),
                    "up" => Quat::from_rotation_z(std::f32::consts::PI),
                    "right" => Quat::from_rotation_z(std::f32::consts::PI / 2.),
                    "down" => Quat::IDENTITY,
                    _ => unimplemented!(),
                };
                p.spawn(SceneBundle {
                    scene: asset_server.load(format!("furniture/{}.glb#Scene0", obj.name)),
                    transform: Transform::default().with_rotation(rot) * base_xform,
                    ..default()
                });
            } else {
                p.spawn(PbrBundle {
                    mesh: meshes.add(Cuboid::new(collider_size, collider_size, collider_size)),
                    material: obj_mat.clone(),
                    ..default()
                });
            }
        });
    }

    // Indicate we should start the game
    commands.insert_resource(ShouldRun);
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
    mut agent_query: Query<(
        Entity,
        &mut Agent,
        &mut KinematicCharacterController,
        &NextAction,
        &Children,
    )>,
    child_query: Query<(Entity, Option<&Name>, Option<&Children>)>,
    mut vis_query: Query<&mut Transform, With<AgentVisuals>>,
    mut anim_query: Query<&mut AnimationPlayer>,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
) {
    for (agent_e, mut agent, mut controller, next_action, children) in agent_query.iter_mut() {
        let dir = next_action.dir;
        let anim_e = get_entity(&agent_e, &["", "", "Root"], &child_query);
        if dir.length_squared() > 0.1 {
            let dir = dir.normalize();
            agent.dir = dir;
            controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
            for child in children.iter() {
                if let Ok(mut xform) = vis_query.get_mut(*child) {
                    xform.look_to(-dir.extend(0.), Vec3::Z);
                    if let Ok(mut anim) = anim_query.get_mut(anim_e.unwrap()) {
                        anim.play_with_transition(
                            asset_server.load("characters/cyborgFemaleA.glb#Animation1"),
                            Duration::from_secs_f32(0.2),
                        )
                        .repeat();
                    }
                    break;
                }
            }
        } else if let Some(anim_e) = anim_e {
            if let Ok(mut anim) = anim_query.get_mut(anim_e) {
                anim.play_with_transition(
                    asset_server.load("characters/cyborgFemaleA.glb#Animation0"),
                    Duration::from_secs_f32(0.2),
                )
                .repeat();
            }
        }
    }
}

/// Returns the entity at this path.
fn get_entity(
    parent: &Entity,
    path: &[&str],
    child_query: &Query<(Entity, Option<&Name>, Option<&Children>)>,
) -> Option<Entity> {
    if path.is_empty() {
        return Some(*parent);
    }
    if let Ok((_, _, Some(children))) = child_query.get(*parent) {
        for child in children {
            let (_, name, _) = child_query.get(*child).unwrap();
            if (name.is_none() && path[0].is_empty()) || (name.unwrap().as_str() == path[0]) {
                let e = get_entity(child, &path[1..], child_query);
                if e.is_some() {
                    return e;
                }
            }
        }
    }
    None
}

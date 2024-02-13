use bevy::{prelude::*, sprite::Anchor};
use bevy_rapier2d::{control::KinematicCharacterController, dynamics::RigidBody, geometry::Collider};
use rand::Rng;

/// Plugin for basic game features, such as moving around and not going through walls.
pub struct GridworldPlugin;

impl Plugin for GridworldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LevelLayout::random(DEFAULT_LEVEL_SIZE))
            .add_systems(Startup, setup_entities);
    }
}

/// Adds playable functionality for `GridworldPlugin`.
pub struct GridworldPlayPlugin;

impl Plugin for GridworldPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_entities_playable)
            .add_systems(
                Update,
                (
                    visualize_agent::<PursuerAgent>(Color::RED),
                    visualize_agent::<PlayerAgent>(Color::GREEN),
                    move_player,
                ),
            );
    }
}

pub const DEFAULT_LEVEL_SIZE: usize = 8;

/// Stores the layout of the level.
#[derive(Resource)]
pub struct LevelLayout {
    /// Stores `true` if a wall exists, `false` for empty spaces. The first element is the top right corner.
    pub walls: Vec<bool>,
    pub size: usize,
}

impl LevelLayout {
    /// Generates a randomized level.
    pub fn random(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            walls: (0..(size * size)).map(|_| rng.gen_bool(0.2)).collect(),
            size,
        }
    }
}

// Indicates the Pursuer agent.
#[derive(Component)]
pub struct PursuerAgent;

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// Sets up all entities in the game.
fn setup_entities(mut commands: Commands, level: Res<LevelLayout>) {
    commands.spawn((
        PursuerAgent,
        Collider::ball(GRID_CELL_SIZE * 0.25),
        RigidBody::KinematicPositionBased,
        KinematicCharacterController::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(20., 0., 0.))),
    ));
    commands.spawn((
        PlayerAgent,
        Collider::ball(GRID_CELL_SIZE * 0.25),
        RigidBody::KinematicPositionBased,
        KinematicCharacterController::default(),
        TransformBundle::from_transform(Transform::from_translation(Vec3::new(40., 30., 0.))),
    ));

    // Set up walls
    for y in 0..level.size {
        for x in 0..level.size {
            if level.walls[y * level.size + x] {
                commands.spawn((
                    Collider::cuboid(GRID_CELL_SIZE / 2., GRID_CELL_SIZE / 2.),
                    TransformBundle::from_transform(Transform::from_translation(
                        Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE,
                    )),
                ));
            }
        }
    }

    // Set up the sides of the game world
    let half_sizes = [GRID_CELL_SIZE / 2., GRID_CELL_SIZE * level.size as f32 / 2.];
    let wall_positions = [-GRID_CELL_SIZE, GRID_CELL_SIZE * level.size as f32];
    let wall_pos_offset = GRID_CELL_SIZE * (level.size - 1) as f32 / 2.;
    for i in 0..4 {
        let positions = [wall_positions[i % 2], wall_pos_offset];
        commands.spawn((
            Collider::cuboid(half_sizes[i / 2], half_sizes[1 - i / 2]),
            TransformBundle::from_transform(Transform::from_translation(Vec3::new(
                positions[i / 2],
                positions[1 - i / 2],
                0.,
            ))),
        ));
    }
}

const GRID_CELL_SIZE: f32 = 25.;

/// Sets up entities for playable mode.
fn setup_entities_playable(mut commands: Commands, level: Res<LevelLayout>) {
    commands.spawn(Camera2dBundle {
        transform: Transform::from_translation(Vec3::new(
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            GRID_CELL_SIZE * ((level.size / 2) as f32 - 0.5),
            0.,
        )),
        ..default()
    });

    for y in 0..level.size {
        for x in 0..level.size {
            if level.walls[y * level.size + x] {
                commands.spawn(SpriteBundle {
                    sprite: Sprite {
                        color: Color::BLACK,
                        custom_size: Some(Vec2::ONE * GRID_CELL_SIZE),
                        ..default()
                    },
                    transform: Transform::from_translation(
                        Vec3::new(x as f32, (level.size - y - 1) as f32, 0.) * GRID_CELL_SIZE,
                    ),
                    ..default()
                });
            }
        }
    }

    // Set up the sides of the game world
    let wall_positions = [
        -GRID_CELL_SIZE * 0.5,
        GRID_CELL_SIZE * (level.size as f32 - 0.5),
    ];
    let wall_pos_offset = GRID_CELL_SIZE * (level.size as f32 / 2. - 0.5);
    let anchors = [
        Anchor::CenterRight,
        Anchor::CenterLeft,
        Anchor::TopCenter,
        Anchor::BottomCenter,
    ];
    for i in 0..4 {
        let positions = [wall_positions[i % 2], wall_pos_offset];
        commands.spawn((SpriteBundle {
            sprite: Sprite {
                color: Color::BLACK,
                custom_size: Some(Vec2::ONE * GRID_CELL_SIZE * level.size as f32 * 2.),
                anchor: anchors[i],
                ..default()
            },
            transform: Transform::from_translation(Vec3::new(
                positions[i / 2],
                positions[1 - i / 2],
                0.,
            )),
            ..default()
        },));
    }
}

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(Commands<'_, '_>, Query<'_, '_, Entity, Added<T>>) {
    move |mut commands: Commands, agent_query: Query<Entity, Added<T>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Sprite {
                    color,
                    custom_size: Some(Vec2::ONE * GRID_CELL_SIZE * 0.25),
                    ..default()
                },
                Handle::<Image>::default(),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
        }
    }
}

const AGENT_SPEED: f32 = GRID_CELL_SIZE * 2.;

/// Allows the player to move the Player agent around.
fn move_player(
    inpt: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut player_query: Query<&mut KinematicCharacterController, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::W) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::S) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::A) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::D) {
        dir.x += 1.;
    }
    let mut controller = player_query.single_mut();
    controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
}

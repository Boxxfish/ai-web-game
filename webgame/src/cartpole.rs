use bevy::prelude::*;
use rand::prelude::*;

pub const GRAVITY: f32 = 9.8;
pub const MASS_CART: f32 = 1.0;
pub const MASS_POLE: f32 = 0.1;
pub const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
pub const LENGTH: f32 = 0.5;
pub const POLE_MASS_LENGTH: f32 = MASS_POLE * LENGTH;
pub const FORCE_MAG: f32 = 10.0;

/// Simulates a cart balancing a pole.
/// Based on the OpenAI gym implementation.
pub struct CartpolePlugin;

impl Plugin for CartpolePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CartpoleState::random())
        .insert_resource(NextAction(2))
            .add_systems(Update, run_sim);
    }
}

/// Adds playable functionality for `CartpolePlugin`.
pub struct CartpolePlayPlugin;

impl Plugin for CartpolePlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_graphics)
            .add_systems(Update, (update_visuals, update_action));
    }
}
/// A cart that will move left and right.
#[derive(Component)]
pub struct Cart;

/// A pole that must be kept upright.
#[derive(Component)]
pub struct Pole;

/// Sets up graphics for the scene.
fn setup_graphics(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    commands
        .spawn((
            Cart,
            SpriteBundle {
                sprite: Sprite {
                    color: Color::RED,
                    custom_size: Some(Vec2::new(100., 10.)),
                    ..default()
                },
                transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
                ..default()
            },
        ))
        .with_children(|p| {
            p.spawn((
                Pole,
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::GREEN,
                        custom_size: Some(Vec2::new(10., 100.)),
                        anchor: bevy::sprite::Anchor::BottomCenter,
                        ..default()
                    },
                    transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
                    ..default()
                },
            ));
        });
}

/// Cart position, cart velocity, pole angle, pole angular velocity.
#[derive(Resource)]
pub struct CartpoleState {
    pub cart_pos: f32,
    pub cart_vel: f32,
    pub pole_angle: f32,
    pub pole_angvel: f32,
}

impl CartpoleState {
    /// Returns a randomized initial state.
    pub fn random() -> Self {
        let low = -0.05;
        let high = 0.05;
        let mut rng = rand::thread_rng();
        Self {
            cart_pos: rng.gen_range(low..high),
            cart_vel: rng.gen_range(low..high),
            pole_angle: rng.gen_range(low..high),
            pole_angvel: rng.gen_range(low..high),
        }
    }
}


/// Runs the simluation.
fn run_sim(mut cart_state: ResMut<CartpoleState>, next_act: Res<NextAction>, time: Res<Time>) {
    let x = cart_state.cart_pos;
    let x_dot = cart_state.cart_vel;
    let theta = cart_state.pole_angle;
    let theta_dot = cart_state.pole_angvel;
    let force = match next_act.0 {
        0 => -FORCE_MAG,
        1 => FORCE_MAG,
        _ => 0.,
    };
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let delta = time.delta_seconds();

    let temp = (force + POLE_MASS_LENGTH * theta_dot.powi(2) * sin_theta) / TOTAL_MASS;
    let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
        / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta.powi(2) / TOTAL_MASS));
    let xacc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

    let x = x + x_dot * delta;
    let x_dot = x_dot + xacc * delta;
    let theta = theta + theta_dot * delta;
    let theta_dot = theta_dot + theta_acc * delta;

    *cart_state = CartpoleState {
        cart_pos: x,
        cart_vel: x_dot,
        pole_angle: theta,
        pole_angvel: theta_dot,
    };
}

/// Updates the visuals in the simulation.
fn update_visuals(
    cart_state: Res<CartpoleState>,
    mut cart_query: Query<&mut Transform, (With<Cart>, Without<Pole>)>,
    mut pole_query: Query<&mut Transform, (With<Pole>, Without<Cart>)>,
) {
    let mut cart_xform = cart_query.single_mut();
    let mut pole_xform = pole_query.single_mut();

    cart_xform.translation.x = cart_state.cart_pos * 100.;
    pole_xform.rotation = Quat::from_rotation_z(cart_state.pole_angle);
}

/// Contains the next action that should be performed in the sim.
#[derive(Resource)]
pub struct NextAction(pub u32);

/// Updates the next action by pressing left or right.
fn update_action(mut next_act: ResMut<NextAction>, inpt: Res<Input<KeyCode>>) {
    next_act.0 = if inpt.pressed(KeyCode::Left) {
        0
    }
    else if inpt.pressed(KeyCode::Right) {
        1
    }
    else {
        2
    };
}
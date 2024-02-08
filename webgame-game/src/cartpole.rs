use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    utils::BoxedFuture,
};
use candle_core::{Module, Tensor};
use candle_nn as nn;
use rand::prelude::*;
use serde::Deserialize;

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
            .add_systems(Update, run_sim)
            .init_asset::<SafeTensorsData>()
            .init_asset_loader::<SafeTensorsDataLoader>();
    }
}

/// Adds playable functionality for `CartpolePlugin`.
pub struct CartpolePlayPlugin;

impl Plugin for CartpolePlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_graphics, init_nn))
            .add_systems(
                Update,
                (
                    update_visuals,
                    update_action_nn,
                    load_weights_into_net::<CartpoleNet>,
                    reset_on_fail_or_success,
                ),
            );
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
        let low = -0.8;
        let high = 0.8;
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

/// Resets the environment if the cart strays too far, or if it successfully balances the pole.
fn reset_on_fail_or_success(mut cart_state: ResMut<CartpoleState>) {
    if cart_state.cart_pos.abs() > 4.
        || (cart_state.pole_angvel.abs() < 0.01 && cart_state.pole_angle.abs() < 0.01)
    {
        *cart_state = CartpoleState::random();
    }
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
/// Add this to allow player control.
fn _update_action(mut next_act: ResMut<NextAction>, inpt: Res<Input<KeyCode>>) {
    next_act.0 = if inpt.pressed(KeyCode::Left) {
        0
    } else if inpt.pressed(KeyCode::Right) {
        1
    } else {
        2
    };
}

/// Updates the next action with the neural network.
/// Add this to allow neural network control.
fn update_action_nn(
    cart_state: Res<CartpoleState>,
    mut next_act: ResMut<NextAction>,
    neural_net: Query<&NNWrapper<CartpoleNet>>,
) {
    if let Ok(neural_net) = neural_net.get_single() {
        if let Some(net) = &neural_net.net {
            let state = Tensor::from_slice(
                &[
                    cart_state.cart_pos,
                    cart_state.cart_vel,
                    cart_state.pole_angle,
                    cart_state.pole_angvel,
                ],
                &[1, 4],
                &candle_core::Device::Cpu,
            )
            .unwrap();
            next_act.0 = net
                .forward(&state)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .argmax(0)
                .unwrap()
                .to_scalar()
                .unwrap();
        }
    }
}

/// Initializes the network.
fn init_nn(asset_server: Res<AssetServer>, mut commands: Commands) {
    commands.spawn(NNWrapper::<CartpoleNet>::with_sftensors(
        asset_server.load("p_net.safetensors"),
    ));
}

/// A component that wraps a neural network.
///
/// Handles loading the safetensors file and initializing the model when ready.
#[derive(Component)]
pub struct NNWrapper<T: LoadableNN> {
    pub weights: Handle<SafeTensorsData>,
    pub net: Option<T>,
}

impl<T: LoadableNN> NNWrapper<T> {
    /// Creates a NNWrapper by passing in a handle to the weights.
    pub fn with_sftensors(sftensors: Handle<SafeTensorsData>) -> Self {
        Self {
            weights: sftensors,
            net: None,
        }
    }
}

/// Allows easily loading weights into the net.
pub trait LoadableNN: std::marker::Send + std::marker::Sync + 'static {
    fn load(vm: nn::VarBuilder) -> candle_core::Result<Self>
    where
        Self: std::marker::Sized;
}

/// The neural network that plays our game.
struct CartpoleNet {
    a_layer1: nn::Linear,
    a_layer2: nn::Linear,
    a_layer3: nn::Linear,
}

impl LoadableNN for CartpoleNet {
    fn load(vm: nn::VarBuilder) -> candle_core::Result<Self> {
        let a_layer1 = nn::linear(4, 64, vm.pp("a_layer1"))?;
        let a_layer2 = nn::linear(64, 64, vm.pp("a_layer2"))?;
        let a_layer3 = nn::linear(64, 2, vm.pp("a_layer3"))?;
        Ok(Self {
            a_layer1,
            a_layer2,
            a_layer3,
        })
    }
}

impl Module for CartpoleNet {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.a_layer1.forward(xs)?.relu()?;
        let xs = self.a_layer2.forward(&xs)?.relu()?;
        let xs = self.a_layer3.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Asset, TypePath, Debug, Deserialize, Clone)]
pub struct SafeTensorsData(pub Vec<u8>);

#[derive(Default)]
pub struct SafeTensorsDataLoader;

#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsError {}

impl AssetLoader for SafeTensorsDataLoader {
    type Asset = SafeTensorsData;
    type Settings = ();
    type Error = SafeTensorsError;
    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a (),
        _load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf).await.unwrap();
            let custom_asset = SafeTensorsData(buf);
            Ok(custom_asset)
        })
    }

    fn extensions(&self) -> &[&str] {
        &["safetensors"]
    }
}

/// Checks if safetensors are loaded and initializes the network if so.
fn load_weights_into_net<T: LoadableNN>(
    mut net_query: Query<&mut NNWrapper<T>>,
    st_assets: Res<Assets<SafeTensorsData>>,
) {
    for mut net in net_query.iter_mut() {
        if net.net.is_none() {
            if let Some(st_data) = st_assets.get(&net.weights) {
                let vb = nn::VarBuilder::from_buffered_safetensors(
                    st_data.0.clone(),
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )
                .unwrap();
                net.net = Some(T::load(vb).expect("Couldn't load model."));
            }
        }
    }
}

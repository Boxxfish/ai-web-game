use bevy::prelude::*;
use candle_core::{DType, Device, Tensor};

use crate::{
    gridworld::LevelLayout,
    models::MeasureModel,
    net::{load_weights_into_net, NNWrapper},
};

/// Plugin for Bayes filtering functionality.
pub struct FilterPlugin;

impl Plugin for FilterPlugin {
    fn build(&self, app: &mut App) {}
}

/// Adds playable functionality to `FilterPlugin`.
pub struct FilterPlayPlugin;

impl Plugin for FilterPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_filter_net).add_systems(
            Update,
            (
                update_filter.run_if(resource_exists::<LevelLayout>),
                load_weights_into_net::<MeasureModel>,
            ),
        );
    }
}

/// Stores data for the filter.
#[derive(Component)]
pub struct BayesFilter {
    pub probs: Tensor,
    pub timer: Timer,
}

impl BayesFilter {
    pub fn new(size: usize) -> Self {
        let probs = (Tensor::ones(&[size, size], DType::F32, &Device::Cpu).unwrap()
            / (size * size) as f64)
            .unwrap();
        Self {
            probs,
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        }
    }
}

/// Initializes the filter.
fn init_filter_net(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        BayesFilter::new(8),
        NNWrapper::<MeasureModel>::with_sftensors(asset_server.load("model.safetensors")),
    ));
}

/// Updates filter probabilities.
fn update_filter(
    mut filter_query: Query<(&mut BayesFilter, &NNWrapper<MeasureModel>)>,
    time: Res<Time>,
    level: Res<LevelLayout>,
) {
    for (mut filter, model) in filter_query.iter_mut() {
        filter.timer.tick(time.delta());
        if filter.timer.just_finished() {
            if let Some(net) = &model.net {
                // Apply motion model
                let kernel = (Tensor::from_slice(
                    &[0., 1., 0., 1., 1., 1., 0., 1., 0.],
                    &[1, 3, 3],
                    &Device::Cpu,
                )
                .unwrap()
                    / 5.)
                    .unwrap()
                    .reshape(&[1, 1, 3, 3])
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let probs = filter
                    .probs
                    .unsqueeze(0)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap()
                    .conv2d(&kernel, 1, 1, 1, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap();
                let probs =
                    (&probs / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64).unwrap();

                // Encode observations

                // Apply measurement model
                let device = Device::Cpu;
                let grid =
                    Tensor::zeros(&[1, 9, level.size, level.size], DType::F32, &device).unwrap();
                let lkhd = net.forward(&grid, None, None).unwrap().squeeze(0).unwrap();
                let probs = (probs * lkhd).unwrap();
                let probs =
                    (&probs / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64).unwrap();

                filter.probs = probs;
            }
        }
    }
}

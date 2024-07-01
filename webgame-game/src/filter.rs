use bevy::prelude::*;

use crate::{
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
        app.add_systems(Startup, init_filter_net)
            .add_systems(Update, load_weights_into_net::<MeasureModel>);
    }
}

/// Stores data for the filter.
#[derive(Component)]
pub struct BayesFilter {
    pub probs: Vec<f32>,
}

/// Initializes the filter network.
fn init_filter_net(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(NNWrapper::<MeasureModel>::with_sftensors(
        asset_server.load("model.safetensors"),
    ));
}

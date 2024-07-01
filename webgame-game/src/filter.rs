use bevy::prelude::*;

/// Plugin for Bayes filtering functionality.
pub struct FilterPlugin;

impl Plugin for FilterPlugin {
    fn build(&self, app: &mut App) {
        ;
    }
}

/// Stores data for the filter.
#[derive(Component)]
pub struct BayesFilter {
    pub probs: Vec<f32>,
}
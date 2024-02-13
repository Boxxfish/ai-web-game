use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    utils::BoxedFuture,
};
use candle_nn as nn;
use serde::Deserialize;

/// Simplifies working with neural networks, particularly loading them.
pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SafeTensorsData>()
            .init_asset_loader::<SafeTensorsDataLoader>();
    }
}

/// A component that wraps a neural network.
///
/// Handles loading the safetensors file and initializing the model when ready.
#[derive(Component)]
pub struct NNWrapper<T: LoadableNN> {
    pub weights: Handle<SafeTensorsData>,
    pub net: Option<T>,
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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

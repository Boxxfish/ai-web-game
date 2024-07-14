use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
        texture::ImageSampler,
    },
};
use candle_core::{DType, Device, Tensor};

use crate::{
    agents::PursuerAgent,
    gridworld::{LevelLayout, GRID_CELL_SIZE},
    models::MeasureModel,
    net::{load_weights_into_net, NNWrapper},
    screens::GameScreen,
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
                init_probs_viewer.run_if(resource_added::<LevelLayout>),
                update_filter.run_if(resource_exists::<LevelLayout>),
                update_probs_viewers,
                load_weights_into_net::<MeasureModel>,
                toggle_viewers,
            ),
        );
    }
}

/// Stores data for the filter.
#[derive(Component)]
pub struct BayesFilter {
    pub probs: Tensor,
    pub size: usize,
}

impl BayesFilter {
    pub fn new(size: usize) -> Self {
        let probs = (Tensor::ones(&[size, size], DType::F32, &Device::Cpu).unwrap()
            / (size * size) as f64)
            .unwrap();
        Self { probs, size }
    }

    pub fn reset(&mut self) {
        self.probs = (Tensor::ones(&[self.size, self.size], DType::F32, &Device::Cpu).unwrap()
            / (self.size * self.size) as f64)
            .unwrap();
    }
}

/// Initializes the filter.
fn init_filter_net(mut commands: Commands) {
    commands.spawn((BayesFilter::new(8),));
}

/// Updates filter probabilities.
#[allow(clippy::too_many_arguments)]
fn update_filter(
    net_query: Query<&NNWrapper<MeasureModel>>,
    mut filter_query: Query<&mut BayesFilter>,
    pursuer_query: Query<&PursuerAgent>,
) {
    for mut filter in filter_query.iter_mut() {
        let model = net_query.single();
        if let Some(net) = &model.net {
            if let Ok(pursuer) = pursuer_query.get_single() {
                if pursuer.obs_timer.just_finished() {
                    if let Some((grid, objs, objs_attn)) = pursuer.observations.as_ref() {
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
                        let probs = (&probs
                            / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64)
                            .unwrap();

                        // Apply measurement model
                        let lkhd = net
                            .forward(
                                &grid.unsqueeze(0).unwrap(),
                                objs.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                                objs_attn.as_ref().map(|t| t.unsqueeze(0).unwrap()).as_ref(),
                            )
                            .unwrap()
                            .squeeze(0)
                            .unwrap();
                        let probs = (probs * lkhd).unwrap();
                        let probs = (&probs
                            / probs.sum_all().unwrap().to_scalar::<f32>().unwrap() as f64)
                            .unwrap();

                        filter.probs = probs;
                    }
                }
            }
        }
    }
}

/// Indicates visuals being used to show filter probabilities.
#[derive(Component)]
struct ProbsViewer {
    pub filter_e: Entity,
}

/// Initializes probs viewers.
fn init_probs_viewer(
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    filter_query: Query<(Entity, &BayesFilter)>,
    level: Res<LevelLayout>,
) {
    for (e, filter) in filter_query.iter() {
        let size = filter.probs.shape().dims()[0];
        let mut img = Image::new(
            Extent3d {
                width: size as u32,
                height: size as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            (0..(size * size))
                .flat_map(|_| [128, 128, 128, 255])
                .collect(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        img.sampler = ImageSampler::nearest();
        let img = images.add(img);
        commands.spawn((
            GameScreen,
            ProbsViewer { filter_e: e },
            PbrBundle {
                mesh: meshes.add(Rectangle::new(
                    GRID_CELL_SIZE * level.size as f32,
                    GRID_CELL_SIZE * level.size as f32,
                )),
                material: materials.add(StandardMaterial {
                    base_color_texture: Some(img),
                    base_color: Color::WHITE.with_a(0.8),
                    unlit: true,
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    ..default()
                }),
                transform: Transform::default()
                    .with_translation(Vec3::new(
                        GRID_CELL_SIZE * (level.size - 1) as f32 / 2.,
                        GRID_CELL_SIZE * (level.size - 1) as f32 / 2.,
                        1.5,
                    ))
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::PI)),
                visibility: Visibility::Hidden,
                ..default()
            },
        ));
    }
}

/// Updates probs viewers.
fn update_probs_viewers(
    filter_query: Query<&BayesFilter>,
    viewer_query: Query<(&ProbsViewer, &Handle<StandardMaterial>)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (viewer, material) in viewer_query.iter() {
        let filter = filter_query.get(viewer.filter_e).unwrap();
        let probs = filter
            .probs
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let data = probs
            .iter()
            .flat_map(|v| [(*v * 255.) as u8, (*v * 255.) as u8, (*v * 255.) as u8, 255])
            .collect::<Vec<_>>();
        if let Some(material) = materials.get_mut(material) {
            if let Some(image) = images.get_mut(material.base_color_texture.as_ref().unwrap()) {
                image.data = data;
                material.base_color_texture = material.base_color_texture.clone();
            }
        }
    }
}

/// Toggles viewers.
fn toggle_viewers(
    inpt: Res<ButtonInput<KeyCode>>,
    mut viewer_query: Query<&mut Visibility, With<ProbsViewer>>,
) {
    if inpt.just_pressed(KeyCode::Space) {
        for mut vis in viewer_query.iter_mut() {
            *vis = match vis.as_ref() {
                Visibility::Inherited | Visibility::Visible => Visibility::Hidden,
                Visibility::Hidden => Visibility::Visible,
            }
        }
    }
}

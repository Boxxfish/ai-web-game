use candle_core::{Device, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use crate::net::LoadableNN;

pub struct BatchNorm1d {}

impl BatchNorm1d {
    pub fn new(num_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        // TODO: Initialize running_mean and running_var
        Ok(Self {})
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub struct BatchNorm2d {}

impl BatchNorm2d {
    pub fn new(num_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {})
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub struct Backbone {
    pub grid_net: nn::Sequential,
    pub pos: Tensor,
}

impl Backbone {
    pub fn new(
        use_pos: bool,
        use_bn: bool,
        channels: usize,
        out_channels: usize,
        size: usize,
        objs_shape: Option<(usize, usize)>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut num_channels = channels;
        if use_pos {
            num_channels += 2;
        }

        // Grid + scalar feature processing
        let mid_channels = 16;
        let grid_net = nn::seq()
            .add(nn::conv2d(
                num_channels,
                mid_channels,
                5,
                nn::Conv2dConfig {
                    padding: 5 / 2,
                    ..Default::default()
                },
                vb.pp("0"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(mid_channels, vb.pp("1"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                mid_channels,
                mid_channels,
                5,
                nn::Conv2dConfig {
                    padding: 5 / 2,
                    ..Default::default()
                },
                vb.pp("3"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(mid_channels, vb.pp("4"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                mid_channels,
                out_channels,
                5,
                nn::Conv2dConfig {
                    padding: 5 / 2,
                    ..Default::default()
                },
                vb.pp("6"),
            )?)
            .add(if use_bn {
                nn::seq().add(BatchNorm2d::new(out_channels, vb.pp("7"))?)
            } else {
                nn::seq()
            })
            .add(nn::Activation::Silu);

        // Positional encoding
        let x_channel = (Tensor::arange(0, size as u32, &Device::Cpu)?
            .unsqueeze(0)?
            .repeat(&[size, 1])?
            .to_dtype(candle_core::DType::F64)?
            / size as f64)?
            .to_dtype(candle_core::DType::F32)?;
        let y_channel = x_channel.t()?;
        let pos = Tensor::stack(&[x_channel, y_channel], 0)?; // Shape: (2, grid_size, grid_size)

        // // Fusion of object features into grid + scalar features
        // if use_objs {
        //     _, obj_dim = objs_shape
        //     n_heads = 4
        //     self.proj = nn.Conv1d(obj_dim, out_channels, 1)
        //     self.attn1 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn1 = nn.BatchNorm1d(out_channels)
        //     self.attn2 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn2 = nn.BatchNorm1d(out_channels)
        //     self.attn3 = nn.MultiheadAttention(
        //         out_channels, n_heads, batch_first=True
        //     )
        //     self.bn3 = nn.BatchNorm1d(out_channels)
        // }

        Ok(Self { grid_net, pos })
    }
}

//         impl Module for Backbone {
//     fn forward(
//         self,
//         grid: Tensor,  // Shape: (batch_size, channels, size, size)
//         objs: Optional[Tensor],  // Shape: (batch_size, max_obj_size, obj_dim)
//         objs_attn_mask: Optional[Tensor],  // Shape: (batch_size, max_obj_size)
//     ) -> Tensor{  // Shape: (batch_size, grid_features_dim, grid_size, grid_size) {
//         // Concat pos encoding to grid
//         device = grid.device
//         if self.use_pos:
//             grid = torch.concat(
//                 [
//                     grid,
//                     self.pos.unsqueeze(0)
//                     .tile([grid.shape[0], 1, 1, 1])
//                     .to(device=device),
//                 ],
//                 dim=1,
//             )
//         grid_features = self.grid_net(
//             grid
//         )  // Shape: (batch_size, grid_features_dim, grid_size, grid_size)

//         if self.use_objs:
//             assert objs is not None
//             assert objs_attn_mask is not None
//             objs = self.proj(objs.permute(0, 2, 1)).permute(
//                 0, 2, 1
//             )  // Shape: (batch_size, max_obj_size, proj_dim)
//             grid_features = grid_features.permute(
//                 0, 2, 3, 1
//             )  // Shape: (batch_size, grid_size, grid_size, grid_features_dim)
//             orig_shape = grid_features.shape
//             grid_features = grid_features.flatten(
//                 1, 2
//             )  // Shape: (batch_size, grid_size * grid_size, grid_features_dim)
//             attns = [self.attn1, self.attn2, self.attn3]
//             bns = [self.bn1, self.bn2, self.bn3]
//             for attn, bn in zip(attns, bns):
//                 attn_grid_features = torch.nan_to_num(
//                     attn(
//                         query=grid_features,
//                         key=objs,
//                         value=objs,
//                         key_padding_mask=objs_attn_mask,
//                     )[0],
//                     0.0,
//                 )  // Sometimes there are no objects, causing NANs
//                 grid_features = grid_features + attn_grid_features
//                 if self.use_bn:
//                     grid_features = bn(grid_features.permute(0, 2, 1)).permute(0, 2, 1)
//                 grid_features = nn.functional.silu(grid_features)
//             grid_features = grid_features.view(orig_shape).permute(
//                 0, 3, 1, 2
//             )  // Shape: (batch_size, grid_features_dim, grid_size, grid_size)

//         return grid_features
//     }
// }

pub struct MeasureModel {
    pub backbone: Backbone,
    pub out_net: nn::Sequential,
}

// Not great, but we don't break these invariants
unsafe impl Send for MeasureModel {}
unsafe impl Sync for MeasureModel {}

impl LoadableNN for MeasureModel {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let channels = 9;
        let size = 8;
        let use_pos = true;
        let objs_shape = None;
        let proj_dim = 32;

        let backbone = Backbone::new(
            use_pos,
            true,
            channels,
            proj_dim,
            size,
            objs_shape,
            vb.pp("backbone"),
        )?;

        // Convert features into liklihood map
        let out_net = nn::seq()
            .add(nn::conv2d(
                proj_dim,
                32,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                vb.pp("0"),
            )?)
            .add(BatchNorm2d::new(32, vb.pp("1"))?)
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                32,
                32,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                vb.pp("3"),
            )?)
            .add(BatchNorm2d::new(32, vb.pp("4"))?)
            .add(nn::Activation::Silu)
            .add(nn::conv2d(
                32,
                1,
                3,
                nn::Conv2dConfig {
                    padding: 3 / 2,
                    ..Default::default()
                },
                vb.pp("6"),
            )?)
            .add(nn::Activation::Sigmoid);

        Ok(MeasureModel { out_net, backbone })
    }
}

//     impl Module for MeasureModel {
//     fn forward(
//         self,
//         grid: Tensor,  // Shape: (batch_size, channels, size, size)
//         objs: Optional[Tensor],  // Shape: (batch_size, max_obj_size, obj_dim)
//         objs_attn_mask: Optional[Tensor],  // Shape: (batch_size, max_obj_size)
//     ) -> Tensor {
//         grid_features = self.backbone(grid, objs, objs_attn_mask)

//         return self.out_net(grid_features).squeeze(
//             1
//         )  // Shape: (batch_size, grid_size, grid_size)
//     }
// }

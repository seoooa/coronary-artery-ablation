# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.inferers import sliding_window_inference

from src.models.proposed.spade import SPADE

__all__ = ["SegResNet", "SegResNetVAE", "SPADESegResNet", "EncoderSPADESegResNet", "PPESegResNet"]


class SegResNet(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.
    SegResNet variant with SPADE normalization only in the decoder path.
    Encoder uses standard ResBlocks without SPADE.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
        label_nc: number of channels in the segmentation map. Defaults to 1.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        label_nc: int = 1,  # New parameter for segmap channels
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        # Initialize parameters
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        # Initialize layers
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers(label_nc)
        self.conv_final = self._make_final_conv(out_channels)
        self.spa_de = SPADE(init_filters, label_nc=label_nc)  # Initialize SPADE with initial filters

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self, label_nc: int):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            spade_layer = SPADE(sample_in_channels // 2, label_nc)  # Initialize SPADE with correct channels
            # Create a list of layers excluding SPADE
            layers = [
                ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
            ]
            up_layer_with_spade = UpLayerWithSPADE(nn.Sequential(*layers), spade_layer)
            up_layers.append(up_layer_with_spade)
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encoder without SPADE normalization"""
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor], segmap: torch.Tensor) -> torch.Tensor:
        """Decoder with SPADE normalization"""
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x, segmap)  # Pass segmap to the custom module

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x, segmap)
        return x


class SegResNetVAE(SegResNet):
    """
    SegResNetVAE based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module contains the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        input_image_size: the size of images to input into the network. It is used to
            determine the in_features of the fc layer in VAE.
        vae_estimate_std: whether to estimate the standard deviations in VAE. Defaults to ``False``.
        vae_default_std: if not to estimate the std, use the default value. Defaults to 0.3.
        vae_nz: number of latent variables in VAE. Defaults to 256.
            Where, 128 to represent mean, and 128 to represent std.
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.
    """

    def __init__(
        self,
        input_image_size: Sequence[int],
        vae_estimate_std: bool = False,
        vae_default_std: float = 0.3,
        vae_nz: int = 256,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: str | tuple = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        label_nc: int = 1,  # New parameter for segmap channels
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            act=act,
            norm=norm,
            use_conv_final=use_conv_final,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode=upsample_mode,
            label_nc=label_nc,
        )

        self.input_image_size = input_image_size
        self.smallest_filters = 16

        zoom = 2 ** (len(self.blocks_down) - 1)
        self.fc_insize = [s // (2 * zoom) for s in self.input_image_size]

        self.vae_estimate_std = vae_estimate_std
        self.vae_default_std = vae_default_std
        self.vae_nz = vae_nz
        self._prepare_vae_modules()
        self.vae_conv_final = self._make_final_conv(in_channels)

    def _prepare_vae_modules(self):
        zoom = 2 ** (len(self.blocks_down) - 1)
        v_filters = self.init_filters * zoom
        total_elements = int(self.smallest_filters * np.prod(self.fc_insize))

        self.vae_down = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, v_filters, self.smallest_filters, stride=2, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.smallest_filters),
            self.act_mod,
        )
        self.vae_fc1 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc2 = nn.Linear(total_elements, self.vae_nz)
        self.vae_fc3 = nn.Linear(self.vae_nz, total_elements)

        self.vae_fc_up_sample = nn.Sequential(
            get_conv_layer(self.spatial_dims, self.smallest_filters, v_filters, kernel_size=1),
            get_upsample_layer(self.spatial_dims, v_filters, upsample_mode=self.upsample_mode),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=v_filters),
            self.act_mod,
        )

    def _get_vae_loss(self, net_input: torch.Tensor, vae_input: torch.Tensor):
        """
        Args:
            net_input: the original input of the network.
            vae_input: the input of VAE module, which is also the output of the network's encoder.
        """
        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)

        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean**2)

            x_vae = z_mean + z_sigma * z_mean_rand

        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        vae_mse_loss = F.mse_loss(net_input, x_vae)
        vae_loss = vae_reg_loss + vae_mse_loss
        return vae_loss

    def forward(self, x, segmap):
        net_input = x
        x, down_x = self.encode(x)
        down_x.reverse()

        vae_input = x
        x = self.decode(x, down_x, segmap)

        if self.training:
            vae_loss = self._get_vae_loss(net_input, vae_input)
            return x, vae_loss

        return x, None


class UpLayerWithSPADE(nn.Module):
    def __init__(self, sequential_block: nn.Sequential, spade_layer: SPADE):
        super(UpLayerWithSPADE, self).__init__()
        self.sequential_block = sequential_block
        self.spade_layer = spade_layer

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x = self.spade_layer(x, segmap)
        x = self.sequential_block(x)
        return x


class SPADEDownBlock(nn.Module):
    """Mirror of UpLayerWithSPADE for encoder path"""
    def __init__(self, sequential_block: nn.Sequential, spade_layer: SPADE):
        super().__init__()
        self.sequential_block = sequential_block
        self.spade_layer = spade_layer

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x = self.spade_layer(x, segmap)
        x = self.sequential_block(x)
        return x


class SPADESegResNet(nn.Module):
    """
    SegResNet variant with symmetric SPADE normalization in both encoder and decoder paths.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        label_nc: int = 1, # New parameter for segmap channels
    ):
        super().__init__()

        # Initialize parameters
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        
        # Initialize layers
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers(label_nc)
        self.up_layers, self.up_samples = self._make_up_layers(label_nc)
        self.conv_final = self._make_final_conv(out_channels)
        self.spa_de = SPADE(init_filters, label_nc=label_nc)  # Initialize SPADE with initial filters

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self, label_nc: int):
        down_layers = nn.ModuleList()
        for i, item in enumerate(self.blocks_down):
            layer_in_channels = self.init_filters * 2**i
            pre_conv = (
                get_conv_layer(self.spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            
            # Create ResBlocks sequence
            blocks = nn.Sequential(*[
                ResBlock(self.spatial_dims, layer_in_channels, norm=self.norm, act=self.act) 
                for _ in range(item)
            ])
            
            # Create SPADE layer
            spade_layer = SPADE(layer_in_channels, label_nc)
            
            # Combine into SPADEDownBlock
            down_layer = nn.Sequential(
                pre_conv,
                SPADEDownBlock(blocks, spade_layer)
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self, label_nc):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        n_up = len(self.blocks_up)
        for i in range(n_up):
            sample_in_channels = self.init_filters * 2 ** (n_up - i)
            spade_layer = SPADE(sample_in_channels // 2, label_nc)
            
            # Create ResBlocks sequence
            blocks = nn.Sequential(*[
                ResBlock(self.spatial_dims, sample_in_channels // 2, norm=self.norm, act=self.act)
            ])
            
            up_layer = UpLayerWithSPADE(blocks, spade_layer)
            up_layers.append(up_layer)
            up_samples.append(
                nn.Sequential(
                    get_conv_layer(self.spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(self.spatial_dims, sample_in_channels // 2, upsample_mode=self.upsample_mode),
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor, segmap: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encoder with SPADE normalization"""
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            if isinstance(down[-1], SPADEDownBlock):
                x = down[0](x)  # Apply pre_conv
                x = down[-1](x, segmap)  # Apply SPADEDownBlock with segmap
            else:
                x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor], segmap: torch.Tensor) -> torch.Tensor:
        """Decoder with SPADE normalization"""
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x, segmap)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x, segmap)
        down_x.reverse()

        x = self.decode(x, down_x, segmap)
        return x


class EncoderSPADESegResNet(nn.Module):
    """
    SegResNet variant with SPADE normalization only in the encoder path.
    Decoder uses standard ResBlocks without SPADE.
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        label_nc: int = 1,  # New parameter for segmap channels
    ):
        super().__init__()

        # Initialize parameters
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        
        # Initialize layers
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers(label_nc)  # SPADE in encoder
        self.up_layers, self.up_samples = self._make_up_layers()  # No SPADE in decoder
        self.conv_final = self._make_final_conv(out_channels)
        self.spa_de = SPADE(init_filters, label_nc=label_nc)  # Initialize SPADE with initial filters

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self, label_nc: int):
        """Create encoder layers with SPADE normalization"""
        down_layers = nn.ModuleList()
        for i, item in enumerate(self.blocks_down):
            layer_in_channels = self.init_filters * 2**i
            pre_conv = (
                get_conv_layer(self.spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            
            # Create ResBlocks sequence
            blocks = nn.Sequential(*[
                ResBlock(self.spatial_dims, layer_in_channels, norm=self.norm, act=self.act) 
                for _ in range(item)
            ])
            
            # Create SPADE layer
            spade_layer = SPADE(layer_in_channels, label_nc)
            
            # Combine into SPADEDownBlock
            down_layer = nn.Sequential(
                pre_conv,
                SPADEDownBlock(blocks, spade_layer)
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        """Create decoder layers with standard ResBlocks (no SPADE)"""
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            
            # Standard ResBlocks without SPADE
            up_layer = nn.Sequential(*[
                ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                for _ in range(blocks_up[i])
            ])
            
            up_layers.append(up_layer)
            up_samples.append(
                nn.Sequential(
                    get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor, segmap: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encoder with SPADE normalization"""
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            if isinstance(down[-1], SPADEDownBlock):
                x = down[0](x)  # Apply pre_conv
                x = down[-1](x, segmap)  # Apply SPADEDownBlock with segmap
            else:
                x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        """Decoder with standard ResBlocks (no SPADE)"""
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)  # Standard ResBlock without segmap

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x, segmap)  # Encoder uses segmap
        down_x.reverse()

        x = self.decode(x, down_x)  # Decoder doesn't use segmap
        return x



class PPEModule(nn.Module):
    def __init__(self, embed_dim, scaling_factor=100.0):
        """
        Parameters:
        -----------
        embed_dim : int
            확장할 채널 수 (첫 번째 컨볼루션 이후의 채널 수)
        scaling_factor : float
            스케일링 팩터
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling_factor = scaling_factor
        
    def to_sin_cos_embedding(self, ppe_tensor):
        """
        PPE를 sinusoidal 임베딩으로 확장
        """
        # 텐서로 변환 및 배치, 채널 차원 추가
        if isinstance(ppe_tensor, np.ndarray):
            ppe_tensor = torch.from_numpy(ppe_tensor).float()
        
        # [H, W, D] -> [1, 1, H, W, D]
        if len(ppe_tensor.shape) == 3:
            ppe_tensor = ppe_tensor.unsqueeze(0).unsqueeze(0)
            
        # 임베딩 차원 준비 (채널 수의 절반만큼 주파수 생성)
        frequencies = 2.0 ** torch.arange(-1, self.embed_dim // 2 - 1).float().to(ppe_tensor.device)
        
        # PPE를 확장하여 각 주파수와 곱함
        ppe_expanded = ppe_tensor * frequencies.view(1, -1, 1, 1, 1) * (2 * math.pi)
        
        # 사인과 코사인 임베딩 계산
        sin_embedding = torch.sin(ppe_expanded)
        cos_embedding = torch.cos(ppe_expanded)
        
        # 사인과 코사인 임베딩 결합하여 원하는 채널 수 생성
        embedding = torch.cat([sin_embedding, cos_embedding], dim=1)
        
        return embedding
        
    def forward(self, ppe, x):
        """
        Parameters:
        -----------
        ppe : torch.Tensor
            PPE 텐서 [H, W, D] 또는 [1, 1, H, W, D]
        x : torch.Tensor
            특징 맵 [B, C, H, W, D]
        """
        # PPE를 sinusoidal 임베딩으로 확장
        ppe_embedding = self.to_sin_cos_embedding(ppe)
        
        # 배치 크기에 맞게 확장
        ppe_embedding = ppe_embedding.expand(x.shape[0], -1, -1, -1, -1)
        
        # 특징 맵에 더하기
        x = x + ppe_embedding.to(x.device)
        
        return x


class PPESegResNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)
        
        # PPE 모듈 초기화
        self.ppe_module = PPEModule(embed_dim=init_filters)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        n_up = len(self.blocks_up)
        for i in range(n_up):
            sample_in_channels = self.init_filters * 2 ** (n_up - i)
            up_layer = nn.Sequential(*[
                ResBlock(self.spatial_dims, sample_in_channels // 2, norm=self.norm, act=self.act)
            ])
            up_layers.append(up_layer)
            up_samples.append(
                nn.Sequential(
                    get_conv_layer(self.spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(self.spatial_dims, sample_in_channels // 2, upsample_mode=self.upsample_mode),
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor, ppe: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 첫 번째 컨볼루션
        x = self.convInit(x)
        
        # PPE 더하기
        x = self.ppe_module(ppe, x)
        
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
            
        return x

    def forward(self, x: torch.Tensor, ppe: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x, ppe)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x

if __name__ == "__main__":
    # Define the input parameters
    img_size = (96, 96, 96)  # Example image size
    in_channels = 1  # Example number of input channels
    out_channels = 2  # Example number of output channels
    init_filters = 8  # Example initial filters
    label_nc = 3  # Example number of segmap channels

    # Create an instance of the model
    model = SPADESegResNet(
        spatial_dims=3,
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=0.0,
        num_groups=8,
        use_conv_final=True,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        upsample_mode=UpsampleMode.NONTRAINABLE,
        label_nc=label_nc
    )

    # Set the model to evaluation mode
    model.eval()

    # Generate dummy input data
    x_in = torch.randn(1, in_channels, *img_size)  # Batch size of 1
    segmap = torch.randn(1, label_nc, *img_size)  # Example segmentation map with multiple channels

    # Define the sliding window parameters
    roi_size = (64, 64, 64)  # Size of the sliding window
    sw_batch_size = 4  # Number of windows to process in parallel
    overlap = 0.25  # Overlap between windows

    # Concatenate x_in and segmap along the channel dimension
    combined_input = torch.cat((x_in, segmap), dim=1)

    # Update the infer_func to handle the combined input
    def infer_func(inputs):
        # Determine the number of channels for x_in from the inputs
        num_channels_x_in = in_channels  # Use the in_channels defined outside
        x_in = inputs[:, :num_channels_x_in, ...]
        segmap = inputs[:, num_channels_x_in:, ...]
        return model(x_in, segmap)

    # Perform sliding window inference with the combined input
    logits = sliding_window_inference(
        inputs=combined_input,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=infer_func,
        overlap=overlap
    )

    # Print the output shape
    print("Sliding window output shape:", logits.shape)
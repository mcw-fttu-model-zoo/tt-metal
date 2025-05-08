# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

from models.experimental.mobilenetv3.reference.mobilenetv3 import Conv2dNormActivation
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    if isinstance(model, Conv2dNormActivation):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


class TtMobileNetV3Conv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        use_shallow_covariant=False,
        activation_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.parameters = parameters
        self.activation_dtype = activation_dtype
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size
        self.shard_layout = shard_layout
        if self.block_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if self.width_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        self.use_shallow_covariant = use_shallow_covariant
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.activation_dtype,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=self.shard_layout,
            input_channels_alignment=16 if self.use_shallow_covariant else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
        )

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt((x.shape[2] // self.batch_size)))
            input_width = int(math.sqrt((x.shape[2] // self.batch_size)))
        [x, [h, w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        x = ttnn.hardswish(x)

        return x, h, w


class TtSqueezeExcitation:
    def __init__(
        self,
        parameters,
        device,
        input_channels: int,
        squeeze_channels: int,
        activation=ttnn.relu,
        scale_activation=ttnn.sigmoid,
        batch_size: int = 1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.activation = activation
        self.scale_activation = scale_activation
        # self.scale_activation = tt_hardsigmoid

        self.avgpool = ttnn.global_avg_pool2d

        self.fc1 = TtMobileNetV3Conv2D(
            device=device,
            parameters=(parameters["fc1"]["weight"], parameters["fc1"]["bias"]),
            input_params=[1, 1, 0, squeeze_channels],
            batch_size=batch_size,
        )

        self.fc2 = TtMobileNetV3Conv2D(
            device=device,
            parameters=(parameters["fc2"]["weight"], parameters["fc2"]["bias"]),
            input_params=[1, 1, 0, input_channels],
            batch_size=batch_size,
        )

    def _scale(self, x):
        pooled = self.avgpool(x)
        x, _, _ = self.fc1(pooled)  # (1, 1, NxHxW, C), (1, 1, 1, 72)
        x = self.activation(x)
        x, _, _ = self.fc2(x)
        x = self.scale_activation(x)
        return x

    def __call__(self, x):
        scale = self._scale(x)
        print(f"scale shape: {scale.shape}, x shape: {x.shape}")
        print(f"scale values: {scale}")
        print(f"x values: {x}")
        assert scale.shape == x.shape, f"scale_padded shape {scale.shape} does not match x shape {x.shape}"

        print(f"scale memory config: {ttnn.get_memory_config(scale)}")
        print(f"x memory config: {ttnn.get_memory_config(x)}")

        scale = ttnn.sharded_to_interleaved(scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        print("scale:", scale)
        print("x:", x)

        res = ttnn.multiply(x, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        return res
        # return x

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


def _make_divisible(v: float, divisor: int, min_value=None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class TtMobileNetV3Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        activation_fn=None,
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
        self.activation_fn = activation_fn
        self.in_channels = in_channels
        self.out_channels = out_channels

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
            in_channels=self.in_channels,
            out_channels=self.out_channels,
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

        x = self.activation_fn(x) if self.activation_fn else x
        return x, h, w


class TtSqueezeExcitation:
    def __init__(
        self,
        parameters,
        device,
        input_channels: int,
        squeeze_channels: int,
        activation=ttnn.relu,
        scale_activation=ttnn.hardsigmoid,
        batch_size: int = 1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.activation = activation
        self.scale_activation = scale_activation

        self.avgpool = ttnn.global_avg_pool2d

        self.fc1 = TtMobileNetV3Conv2D(
            device=device,
            in_channels=input_channels,
            out_channels=squeeze_channels,
            parameters=(parameters["fc1"]["weight"], parameters["fc1"]["bias"]),
            input_params=[1, 1, 0, squeeze_channels],
            batch_size=batch_size,
            activation_fn=activation,
        )

        self.fc2 = TtMobileNetV3Conv2D(
            device=device,
            in_channels=squeeze_channels,
            out_channels=input_channels,
            parameters=(parameters["fc2"]["weight"], parameters["fc2"]["bias"]),
            input_params=[1, 1, 0, input_channels],
            batch_size=batch_size,
            activation_fn=activation,
        )

    def _scale(self, x):
        pooled = self.avgpool(x)
        x, _, _ = self.fc1(pooled)
        x = self.activation(x)
        x, _, _ = self.fc2(x)
        x = self.scale_activation(x)
        return x

    def __call__(self, x):
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        scale = self._scale(x)
        scale = ttnn.sharded_to_interleaved(scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        res = ttnn.multiply(x, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        return res


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class TtInvertedResidual:
    def __init__(
        self,
        config: InvertedResidualConfig,
        model_params,
        device,
        batch_size,
    ):
        if not (1 <= config.stride <= 2):
            raise ValueError("illegal stride value")
        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels
        self.device = device

        activation_layer = ttnn.hardswish if config.use_hs else ttnn.relu

        self.conv = []
        idx = 0

        # 1. Expand conv
        if config.expanded_channels != config.input_channels:
            print("Control inside if expanded_channels != input_channels")
            print("activation_layer: ", activation_layer)
            self.conv.append(
                TtMobileNetV3Conv2D(
                    device=device,
                    in_channels=config.input_channels,
                    out_channels=config.expanded_channels,
                    parameters=(
                        model_params["block"][idx]["conv"]["weight"],
                        model_params["block"][idx]["conv"]["bias"],
                    ),
                    input_params=[1, 1, 0, config.expanded_channels],
                    batch_size=batch_size,
                    activation_fn=activation_layer,
                )
            )
            idx += 1

        stride = 1 if config.dilation > 1 else config.stride
        print("stride: ", stride)

        # 2. Depthwise conv
        self.conv.append(
            TtMobileNetV3Conv2D(
                device=device,
                in_channels=config.expanded_channels,
                out_channels=config.expanded_channels,
                parameters=(model_params["block"][idx]["conv"]["weight"], model_params["block"][idx]["conv"]["bias"]),
                input_params=[config.kernel, stride, config.kernel // 2, config.expanded_channels],
                batch_size=batch_size,
                groups=config.expanded_channels,
                activation_fn=activation_layer,
            )
        )
        idx += 1

        # 3. Optional SE
        if config.use_se:
            squeeze_channels = _make_divisible(config.expanded_channels // 4, 8)
            self.conv.append(
                TtSqueezeExcitation(
                    parameters=model_params["block"][idx],
                    device=device,
                    input_channels=config.expanded_channels,
                    squeeze_channels=squeeze_channels,
                    activation=ttnn.relu,
                    scale_activation=ttnn.hardsigmoid,
                    batch_size=batch_size,
                )
            )
            idx += 1

        # 4. Project conv
        self.conv.append(
            TtMobileNetV3Conv2D(
                device=device,
                in_channels=config.expanded_channels,
                out_channels=config.out_channels,
                parameters=(model_params["block"][idx]["conv"]["weight"], model_params["block"][idx]["conv"]["bias"]),
                input_params=[1, 1, 0, config.out_channels],
                batch_size=batch_size,
                activation_fn=None,  # No activation function
            )
        )

        self.out_channels = config.out_channels
        self._is_cn = config.stride > 1

    def __call__(self, x):
        identity = x

        for i, conv in enumerate(self.conv):
            out = conv(x)
            x = out[0] if isinstance(out, tuple) else out

            print(f"x shape for {i}: ", x.shape)

        identity = ttnn.to_layout(identity, ttnn.TILE_LAYOUT)
        identity = ttnn.sharded_to_interleaved(identity, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        if self.use_res_connect:
            x = ttnn.reshape(x, identity.shape)
            x += identity

        return x

import ttnn
import torch
import pytest

from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.mobilenetv3.reference.torch_utils import (
    Conv2dNormActivation as TVConv2dNormActivation,
    SqueezeExcitation as TVSqueezeExcitation,
)

# from torchvision.ops.misc import Conv2dNormActivation as TVConv2dNormActivation
# from torchvision.ops.misc import SqueezeExcitation as TVSqueezeExcitation

from models.experimental.mobilenetv3.tt.common_utils import (
    TtMobileNetV3Conv2D,
    TtSqueezeExcitation,
    TtInvertedResidual,
    InvertedResidualConfig,
)
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from models.experimental.mobilenetv3.reference.mobilenetv3 import mobilenet_v3_large as tv_mv3

# from torchvision.models.mobilenetv3 import mobilenet_v3_large as tv_mv3

from ttnn.model_preprocessing import preprocess_model_parameters

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


def fold_batch_norm2d_into_conv2d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
    weight = conv.weight.data
    running_mean = bn.running_mean
    running_var = bn.running_var.data
    eps = bn.eps
    scale = bn.weight.data
    shift = bn.bias.data
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32)
    return weight, bias


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, TVSqueezeExcitation):
        # Preprocess fc1
        fc1_weight = model.fc1.weight
        fc1_bias = model.fc1.bias

        parameters["fc1"] = {
            "weight": ttnn.from_torch(fc1_weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(fc1_bias.reshape(1, 1, 1, -1), dtype=ttnn.float32),
        }

        # Preprocess fc2
        fc2_weight = model.fc2.weight
        fc2_bias = model.fc2.bias

        parameters["fc2"] = {
            "weight": ttnn.from_torch(fc2_weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(fc2_bias.reshape(1, 1, 1, -1), dtype=ttnn.float32),
        }

    elif isinstance(model, TVConv2dNormActivation):
        conv = model[0]  # Conv2d
        bn = model[1]  # BatchNorm2d

        folded_weight, folded_bias = fold_batch_norm2d_into_conv2d(conv, bn)
        parameters["conv"] = {"weight": folded_weight, "bias": folded_bias.reshape(1, 1, 1, -1)}  # required shape
    return parameters


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "torch_input, layer",
    [
        (torch.rand(1, 3, 224, 224), "model.0"),  # 0
    ],
)
def test_conv(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_model_output = sub_module(torch_input)
    kernel_size = 3
    stride = 2
    padding = 1
    out_channels = 16

    input_params = (kernel_size, stride, padding, out_channels)
    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        in_channels=ttnn_input.shape[3],
        out_channels=out_channels,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=input_params,
        batch_size=ttnn_input.shape[0],
        activation_fn=ttnn.hardswish,
    )
    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)
    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )
    ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))
    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, 0.999))  # pcc=0.93 if use_pretrained_weight else


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "torch_input, layer", [(torch.rand(1, 72, 28, 28), "features.4.block.2")]  # SE block inside InvertedResidual[4]
)
def test_se(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[4].block[2]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    torch_model_output = sub_module(torch_input)

    ttnn_model = TtSqueezeExcitation(
        device=device,
        parameters=parameters,
        input_channels=72,
        squeeze_channels=24,
        activation=ttnn.relu,
        scale_activation=ttnn.hardsigmoid,
        batch_size=ttnn_input.shape[0],
    )
    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )
    ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))
    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, 0.999))


# test for the inverted residual submodule
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "torch_input, layer",
    [
        (torch.rand(1, 16, 112, 112), "features.4.block.2"),  # InvertedResidual[4]
    ],
)
def test_inverted_residual(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[1]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    torch_model_output = sub_module(torch_input)

    config = InvertedResidualConfig(
        input_channels=16,
        kernel=3,
        expanded_channels=16,
        out_channels=16,
        use_se=False,
        activation="RE",
        stride=1,
        dilation=1,
        width_mult=1.0,
    )

    ttnn_model = TtInvertedResidual(
        config=config,
        model_params=parameters,
        device=device,
        batch_size=ttnn_input.shape[0],
    )
    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )
    ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))
    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, 0.99))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 16, 112, 112), "features.1.block.0")])
def test_inverted_residual_conv0(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)

    sub_module = tv_model.features[1].block[0]
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute((0, 2, 3, 1))  # NCHW → NHWC
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_model_output = sub_module(torch_input)
    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        in_channels=ttnn_input.shape[3],
        out_channels=16,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=(3, 1, 1, 16),
        batch_size=ttnn_input.shape[0],
        groups=16,
        activation_fn=ttnn.relu,
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    ).permute((0, 3, 1, 2))

    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, pcc=0.999))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 16, 112, 112), "features.1.block.1")])
def test_inverted_residual_conv1(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)

    sub_module = tv_model.features[1].block[1]  # projection conv
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )
    ttnn_input = torch_input.permute((0, 2, 3, 1))  # NCHW → NHWC
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_model_output = sub_module(torch_input)

    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        in_channels=ttnn_input.shape[3],
        out_channels=16,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=(1, 1, 0, 16),  # kernel, stride, padding, out_channels
        batch_size=ttnn_input.shape[0],
        groups=1,
        activation_fn=None,  # no activation for projection
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    ).permute((0, 3, 1, 2))

    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, pcc=0.999))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 24, 56, 56), "features.4")])
def test_inverted_residual_block4(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)

    block_id = 4
    sub_module = tv_model.features[block_id]

    # Preprocess weights
    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module, custom_preprocessor=custom_preprocessor
    )

    print("sub_module", sub_module)
    print("parameters", parameters)

    # # Convert input
    ttnn_input = torch_input.permute(0, 2, 3, 1)  # NCHW → NHWC
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    # # # Run Torch reference
    torch_model_output = sub_module(torch_input)

    config = InvertedResidualConfig(
        input_channels=24,
        kernel=5,
        expanded_channels=72,
        out_channels=40,
        use_se=True,
        activation="RE",
        stride=2,
        dilation=1,
        width_mult=1.0,
    )

    ttnn_model = TtInvertedResidual(
        config=config,
        model_params=parameters,
        device=device,
        batch_size=ttnn_input.shape[0],
    )

    ttnn_model_output = ttnn_model(ttnn_input)
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    print("torch_model_output.shape", torch_model_output.shape)
    print("ttnn_model_output.shape", ttnn_model_output.shape)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )

    ttnn_model_output = ttnn_model_output.permute(0, 3, 1, 2)

    print("torch_model_output", torch_model_output[0, 0, 0, :10])
    print("ttnn_model_output", ttnn_model_output[0, 0, 0, :10])

    pretrained_weight = True
    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, 0.93 if pretrained_weight else 0.999))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 24, 56, 56), "features.4.block.0")])
def test_inverted_residual_block4_conv0(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[4].block[0]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    torch_model_output = sub_module(torch_input)

    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        in_channels=ttnn_input.shape[3],
        out_channels=72,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=(1, 1, 0, 72),
        batch_size=ttnn_input.shape[0],
        groups=1,
        activation_fn=ttnn.relu,
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )

    ttnn_model_output = ttnn_model_output.permute(0, 3, 1, 2)

    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, pcc=0.99))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 72, 56, 56), "features.4.block.0")])
def test_inverted_residual_block4_conv1(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[4].block[1]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    torch_model_output = sub_module(torch_input)

    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        in_channels=ttnn_input.shape[3],
        out_channels=72,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=(5, 2, 2, 72),
        batch_size=ttnn_input.shape[0],
        groups=72,
        activation_fn=ttnn.relu,
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )

    ttnn_model_output = ttnn_model_output.permute(0, 3, 1, 2)

    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, pcc=0.99))


# # test for the InvertedResidual[11] submodule
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize("torch_input, layer", [(torch.rand(1, 80, 28, 28), "features.11")])
def test_inverted_residual_block11(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()
    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    sub_module = tv_model.features[11]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )

    print("sub_module", sub_module)
    print("parameters", parameters)

    ttnn_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    torch_model_output = sub_module(torch_input)

    config = InvertedResidualConfig(
        input_channels=80,
        kernel=3,
        expanded_channels=480,
        out_channels=112,
        use_se=True,
        activation="HS",
        stride=1,
        dilation=1,
        width_mult=1.0,
    )

    ttnn_model = TtInvertedResidual(
        config=config,
        model_params=parameters,
        device=device,
        batch_size=ttnn_input.shape[0],
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )

    ttnn_model_output = ttnn_model_output.permute(0, 3, 1, 2)

    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, pcc=0.999))

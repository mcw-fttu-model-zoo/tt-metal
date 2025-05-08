import ttnn
import torch
import pytest

from models.utility_functions import run_for_wormhole_b0
from models.utility_functions import disable_persistent_kernel_cache
from torchvision.ops.misc import Conv2dNormActivation as TVConv2dNormActivation
from torchvision.ops.misc import SqueezeExcitation as TVSqueezeExcitation

from models.experimental.mobilenetv3.tt.common_utils import TtMobileNetV3Conv2D, TtSqueezeExcitation
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large as tv_mv3

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
    print("model", model)
    print(f"Type of model: {type(model)}")
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
        print("parameters updated!")

    elif isinstance(model, TVConv2dNormActivation):
        conv = model[0]  # Conv2d
        bn = model[1]  # BatchNorm2d

        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
        parameters["conv"] = {}
        parameters["conv"]["weight"] = weight
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"]["bias"] = bias

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
    print("parameters obtained:", parameters)

    ttnn_input = torch_input.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_model_output = sub_module(torch_input)

    # ttnn output
    # Conv2dNormActivation(
    #     3,
    #     firstconv_output_channels,
    #     kernel_size=3,
    #     stride=2,
    #     norm_layer=norm_layer,
    #     activation_layer=nn.Hardswish,
    # )
    kernel_size = 3
    stride = 2
    padding = 1
    out_channels = 16

    input_params = (kernel_size, stride, padding, out_channels)
    ttnn_model = TtMobileNetV3Conv2D(
        device=device,
        parameters=(parameters["conv"]["weight"], parameters["conv"]["bias"]),
        input_params=input_params,
        batch_size=ttnn_input.shape[0],
    )

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)
    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )
    ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    # Print the shapes of the outputs
    logger.info(f"torch_model_output shape: {torch_model_output.shape}")
    logger.info(f"ttnn_model_output shape: {ttnn_model_output.shape}")

    # Print the 10 elements of the outputs
    logger.info(f"torch_model_output: {torch_model_output[0, 0, 0, :10]}")
    logger.info(f"ttnn_model_output: {ttnn_model_output[0, 0, 0, :10]}")

    # use_pretrained_weight = True
    logger.info(assert_with_pcc(torch_model_output, ttnn_model_output, 0.999))  # pcc=0.93 if use_pretrained_weight else


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
@pytest.mark.parametrize(
    "torch_input, layer", [(torch.rand(1, 72, 28, 28), "features.4.block.2")]  # SE block inside InvertedResidual[4]
)
def test_se(device, torch_input, layer, use_program_cache, reset_seeds):
    disable_persistent_kernel_cache()

    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # print("tv_model",tv_model)
    sub_module = tv_model.features[4].block[2]

    print("sub_module", sub_module)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: sub_module,
        custom_preprocessor=custom_preprocessor,
    )
    print("parameters obtained:", parameters)

    ttnn_input = torch_input.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)

    torch_model_output = sub_module(torch_input)
    print("torch_model_output shape:", torch_model_output.shape)

    ttnn_model = TtSqueezeExcitation(
        device=device,
        parameters=parameters,
        input_channels=72,
        squeeze_channels=24,
        activation=ttnn.relu,
        scale_activation=ttnn.sigmoid,
        batch_size=ttnn_input.shape[0],
    )

    print("ttnn_model", ttnn_model)

    ttnn_model_output = ttnn_model(ttnn_input)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    ttnn_model_output = ttnn_model_output.reshape(
        1, torch_model_output.shape[2], torch_model_output.shape[3], ttnn_model_output.shape[-1]
    )
    ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    # # Print the shapes of the outputs
    logger.info(f"torch_model_output : {torch_model_output}")
    logger.info(f"ttnn_model_output: {ttnn_model_output}")

    # # Print the 10 elements of the outputs
    # logger.info(f"torch_model_output: {torch_model_output[0, 0, 0, :10]}")
    # logger.info(f"ttnn_model_output: {ttnn_model

    use_pretrained_weight = True
    logger.info(
        assert_with_pcc(torch_model_output, ttnn_model_output, 0.97 if use_pretrained_weight else 0.999)
    )  # pcc=0.93 if use_pretrained_weight else 0.999


# import numpy as np

# @pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)

# def test_hardsigmoid(device, use_program_cache, reset_seeds):
#     # Simulate input to FC2 output: small positive value
#     torch_input = torch.tensor([[[[0.0088]]]], dtype=torch.float32)

#     # Torch reference
#     torch_hardsigmoid = torch.nn.Hardsigmoid()
#     torch_output = torch_hardsigmoid(torch_input)

#     # TTNN test
#     ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
#     ttnn_output = ttnn.hardsigmoid(ttnn_input)
#     ttnn_output = ttnn.to_torch(ttnn_output)

#     # Print outputs
#     logger.info(f"Torch input: {torch_input}")
#     logger.info(f"Torch hardsigmoid output: {torch_output}")
#     logger.info(f"TTNN hardsigmoid output: {ttnn_output}")

#     # Compare using PCC
#     assert_with_pcc(torch_output, ttnn_output, pcc=0.999)

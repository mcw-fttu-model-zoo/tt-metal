import torch
from models.experimental.mobilenetv3.reference.mobilenetv3 import mobilenet_v3_large as ref_mv3
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large as tv_mv3


def test_model_equivalence(tol: float = 1e-5):
    dummy_input = torch.randn(1, 3, 224, 224)

    tv_model = tv_mv3(weights=MobileNet_V3_Large_Weights.DEFAULT)
    state_dict = tv_model.state_dict()
    tv_model.eval()

    ref_model = ref_mv3()
    ref_model.load_state_dict(state_dict)
    ref_model.eval()

    with torch.no_grad():
        ref_output = ref_model(dummy_input)
        tv_output = tv_model(dummy_input)

    # PRINT OUTPUTS shape
    print("Reference model output shape:", ref_output.shape)
    print("Torchvision model output shape:", tv_output.shape)

    # print first 10 elements of the output
    print("Reference model output (first 10):", ref_output[0, :10])
    print("Torchvision model output (first 10):", tv_output[0, :10])

    max_diff = (ref_output - tv_output).abs().max().item()
    print("Max absolute difference:", max_diff)

    are_close = torch.allclose(ref_output, tv_output, rtol=1e-4, atol=tol)
    print("Outputs close within tolerance:", are_close)

    assert are_close, f"Model outputs differ! Max diff: {max_diff}"


if __name__ == "__main__":
    test_model_equivalence()

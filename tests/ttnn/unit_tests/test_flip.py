import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_tensor, dim",
    [
        # (torch.arange(10), 0),  # 1D case, simple reverse
        # 2D Cases
        (torch.arange(12).reshape([4, 3]), [1, 0]),  # Flip rows
        (torch.arange(12).reshape([4, 3]), [0, 1]),  # Flip columns
        (torch.arange(6).reshape([2, 3]), [0, 1]),  # Smaller 2D case, flip rows
        (torch.arange(6).reshape([2, 3]), [0, 1]),  # Smaller 2D case, flip columns
        (torch.arange(16).reshape([4, 4]), [0, 1]),  # Flip rows in square matrix
        (torch.arange(16).reshape([4, 4]), [0, 1]),  # Flip columns in square matrix
        (torch.arange(25).reshape([5, 5]), [0, 1]),  # Flip rows in larger square matrix
        (torch.arange(25).reshape([5, 5]), [0, 1]),  # Flip columns in larger square matrix
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], [0, -1]),  # 4d Check
        ([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]], [0, 2, -1]),  # 5d Check
        # 3D Cases (Flipping at least two dimensions)
        (torch.arange(24).reshape([2, 4, 3]), [0, 1]),  # Flip first and second dimensions
        (torch.arange(24).reshape([2, 4, 3]), [1, 2]),  # Flip second and third dimensions
        (torch.arange(30).reshape([3, 5, 2]), [0, 2]),  # Flip first and third dimensions
        (torch.arange(30).reshape([3, 5, 2]), [0, 1, 2]),  # Flip all dimensions
        # 4D Cases (Flipping at least two dimensions)
        (torch.arange(48).reshape([2, 3, 4, 2]), [0, 1]),  # Flip first and second dimensions
        (torch.arange(48).reshape([2, 3, 4, 2]), [1, 2]),  # Flip second and third dimensions
        (torch.arange(48).reshape([2, 3, 4, 2]), [2, 3]),  # Flip third and fourth dimensions
        (torch.arange(48).reshape([2, 3, 4, 2]), [0, 2, 3]),  # Flip first, third, and fourth dimensions
        (torch.arange(48).reshape([2, 3, 4, 2]), [-1, 2, 0]),  # Flip all dimensions
        # # Large Tensors (Flipping at least two dimensions)
        (torch.arange(100).reshape([10, 10]), [0, 1]),  # Large 2D matrix, flip rows and columns
        (torch.arange(120).reshape([5, 4, 6]), [0, 1]),  # Flip first and second dimensions
        (torch.arange(120).reshape([5, 4, 6]), [1, 2]),  # Flip second and third dimensions
        (torch.arange(120).reshape([5, 4, 6]), [0, 2]),  # Flip first and third dimensions
        (torch.arange(120).reshape([5, 4, 6]), [0, 1, 2]),  # Flip all dimensions
        # Edge Cases (Flipping at least two dimensions)
        (torch.arange(8).reshape([2, 2, 2]), [0, 1]),  # 3D tensor, flip first and second dimensions
        (torch.arange(8).reshape([2, 2, 2]), [1, 2]),  # 3D tensor, flip second and third dimensions
        (torch.arange(8).reshape([2, 2, 2]), [0, 2]),  # 3D tensor, flip first and third dimensions
        (torch.arange(8).reshape([2, 2, 2]), [0, 1, 2]),  # Flip all dimensions
        (
            torch.tensor(
                [
                    [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
                    [[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]],
                ]
            ),
            [-1],
        ),
        (
            [
                [[[[[5, 6], [7, 8]], [[9, 10], [11, 12]]], [[[13, 14], [15, 16]], [[17, 18], [19, 20]]]]],
                [[[[[21, 22], [23, 24]], [[25, 26], [27, 28]]], [[[29, 30], [31, 32]], [[33, 34], [35, 36]]]]],
            ],
            [5, 0],
        ),
        (
            [
                [
                    [[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]],
                    [[[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]]],
                ]
            ],
            [0, 4, -2],
        ),  # 6d Check
    ],
)
def test_ttnn_flip(device, input_tensor, dim):
    tensor = torch.tensor(input_tensor, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(tensor)

    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    torch_flipped = torch.flip(tensor, dim)
    ttnn_flipped = ttnn.flip(ttnn_tensor, dim)

    ttnn_flipped_torch = ttnn.to_torch(ttnn_flipped)

    print("Torch Output:", torch_flipped)
    print("TTNN Output:", ttnn_flipped_torch)

    assert_with_pcc(torch_flipped, ttnn_flipped_torch)
    assert (
        torch_flipped.shape == ttnn_flipped_torch.shape
    ), f"Shape mismatch: {torch_flipped.shape} vs {ttnn_flipped_torch.shape}"

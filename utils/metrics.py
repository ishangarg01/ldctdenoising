# utils/metrics.py

import numpy as np
import torch
# Use scikit-image for PSNR calculation (standard implementation)
from skimage.metrics import peak_signal_noise_ratio as calculate_skimage_psnr
# Use pytorch-msssim for SSIM calculation (PyTorch implementation)
from pytorch_msssim import ssim as calculate_pytorch_ssim
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tensor2numpy(tensor, rgb_range=1.0):
    """
    Converts a PyTorch tensor to a NumPy array.
    Assumes the tensor is in (C, H, W) format and in the range [0, rgb_range].
    Converts to (H, W, C) format and scales to [0, 255] uint8.

    Args:
        tensor (torch.Tensor): Input tensor.
        rgb_range (float): The maximum value in the tensor (e.g., 1.0 or 255.0).

    Returns:
        numpy.ndarray: Converted NumPy array in uint8 format, shape (H, W, C).
    """
    # Move to CPU and detach from graph
    tensor = tensor.detach().cpu()
    # Scale to [0, 255] and convert to uint8
    # Clamp values to the valid range [0, rgb_range] before scaling
    img_np = tensor.mul(255.0 / rgb_range).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return img_np

def calculate_psnr(img1, img2, data_range=255.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor or numpy.ndarray): First image.
        img2 (torch.Tensor or numpy.ndarray): Second image.
        data_range (float): The range of the image data (e.g., 255.0 for uint8, 1.0 for float [0,1]).

    Returns:
        float: PSNR value.
    """
    # Ensure inputs are NumPy arrays
    if isinstance(img1, torch.Tensor):
        # Convert from (C, H, W) float [0,1] to (H, W, C) uint8 [0,255]
        img1_np = tensor2numpy(img1, rgb_range=1.0)
    else: # Assume NumPy array
        img1_np = img1

    if isinstance(img2, torch.Tensor):
         # Convert from (C, H, W) float [0,1] to (H, W, C) uint8 [0,255]
        img2_np = tensor2numpy(img2, rgb_range=1.0)
    else: # Assume NumPy array
        img2_np = img2

    # Calculate PSNR using scikit-image's function
    # Assumes data_range is 255 for uint8 inputs
    psnr_val = calculate_skimage_psnr(img1_np, img2_np, data_range=255.0) # skimage expects data_range of the input arrays
    return psnr_val


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculates Structural Similarity Index Measure (SSIM) between two images.

    Args:
        img1 (torch.Tensor): First image tensor (N, C, H, W) or (C, H, W).
        img2 (torch.Tensor): Second image tensor (N, C, H, W) or (C, H, W).
        data_range (float): The range of the image data (e.g., 1.0 for float [0,1], 255.0 for uint8).

    Returns:
        float: SSIM value.
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError("Inputs to calculate_ssim must be PyTorch tensors.")

    # Add batch dimension if missing
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    # Calculate SSIM using pytorch-msssim's function
    # Assumes data_range is 1.0 for float [0,1] tensors, or 255.0 for uint8 [0,255] tensors
    # We are using float [0,1] tensors from dataset, so data_range=1.0 is correct.
    ssim_val = calculate_pytorch_ssim(img1, img2, data_range=data_range, size_average=True) # size_average=True returns a scalar

    return ssim_val.item() # Return as a standard Python float


# Example Usage (for testing the metrics)
if __name__ == '__main__':
    print("--- Testing metrics.py ---")

    # Create dummy image tensors (float, range [0, 1])
    dummy_img1_tensor = torch.rand(3, 256, 256)
    # Create a slightly different image for comparison
    dummy_img2_tensor = dummy_img1_tensor + torch.randn(3, 256, 256) * 0.05
    dummy_img2_tensor = torch.clamp(dummy_img2_tensor, 0, 1) # Clamp to [0, 1] range

    # Create identical images for perfect score
    dummy_img_identical1 = torch.rand(3, 256, 256)
    dummy_img_identical2 = dummy_img_identical1.clone()

    print("\nTesting with float [0, 1] tensors:")
    # Calculate PSNR (expects uint8 [0, 255] internally, so conversion is needed)
    psnr_val = calculate_psnr(dummy_img1_tensor, dummy_img2_tensor, data_range=1.0)
    print(f"PSNR between dummy_img1 and dummy_img2: {psnr_val:.4f}")

    # Calculate SSIM (expects float [0, 1] or uint8 [0, 255] depending on data_range)
    ssim_val = calculate_ssim(dummy_img1_tensor, dummy_img2_tensor, data_range=1.0)
    print(f"SSIM between dummy_img1 and dummy_img2: {ssim_val:.4f}")

    # Test PSNR with identical images
    psnr_identical = calculate_psnr(dummy_img_identical1, dummy_img_identical2, data_range=1.0)
    print(f"PSNR between identical images: {psnr_identical:.4f}") # Should be very high or infinity

    # Test SSIM with identical images
    ssim_identical = calculate_ssim(dummy_img_identical1, dummy_img_identical2, data_range=1.0)
    print(f"SSIM between identical images: {ssim_identical:.4f}") # Should be 1.0


    # Test with dummy NumPy arrays (uint8, range [0, 255])
    dummy_img1_np = (dummy_img1_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    dummy_img2_np = (dummy_img2_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    print("\nTesting with uint8 [0, 255] NumPy arrays:")
    # Calculate PSNR (expects uint8 [0, 255] input, data_range=255.0)
    psnr_val_np = calculate_psnr(dummy_img1_np, dummy_img2_np, data_range=255.0)
    print(f"PSNR between dummy_img1_np and dummy_img2_np: {psnr_val_np:.4f}")

    # Note: calculate_ssim from pytorch-msssim primarily works on tensors.
    # If you needed to calculate SSIM on NumPy arrays, you'd convert them to tensors first.
    # For consistency with the project pipeline (which uses tensors), we keep SSIM tensor-based.
    print("SSIM calculation directly on NumPy arrays is not supported by the chosen library.")
    # Example of converting NumPy to Tensor for SSIM:
    # dummy_img1_tensor_from_np = torch.from_numpy(dummy_img1_np).permute(2, 0, 1).float() / 255.0
    # dummy_img2_tensor_from_np = torch.from_numpy(dummy_img2_np).permute(2, 0, 1).float() / 255.0
    # ssim_val_np_to_tensor = calculate_ssim(dummy_img1_tensor_from_np, dummy_img2_tensor_from_np, data_range=1.0)
    # print(f"SSIM (NumPy converted to Tensor): {ssim_val_np_to_tensor:.4f}")


    print("--- Metrics test complete ---")

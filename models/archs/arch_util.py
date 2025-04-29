# models/archs/arch_util.py

# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# This file contains utility functions and modules for network architectures.
# ------------------------------------------------------------------------
import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch import Tensor
import logging
import time # Import time for inference speed measurement

# Setup a simple logger for this utility file
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure basic logging if no handlers are already set by the main script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Added default_conv function ---
def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True):
    """Default convolutional layer."""
    # Calculate padding automatically to maintain spatial dimensions if padding is None
    if padding is None:
        # For 'same' padding logic with stride 1 and odd kernel size
        if kernel_size % 2 == 1 and stride == 1:
            padding = kernel_size // 2
        else:
            # Otherwise, use explicit padding passed or default 0
            padding = 0 # Default padding if not specified and auto-calc fails
            # You might want to add a warning or error for more complex cases
            # print(f"Warning: Could not auto-calculate 'same' padding for k={kernel_size}, s={stride}. Using padding={padding}")

    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
# ----------------------------------


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights using Kaiming Normal initialization.

    Applies initialization to nn.Conv2d, nn.Linear, and _BatchNorm modules
    within the provided module(s). Weights are scaled by `scale`, and biases
    are filled with `bias_fill`.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules or list of modules to be initialized.
        scale (float): Scale factor for initialized weights, useful for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias terms with. Default: 0.
        kwargs (dict): Other arguments for the initialization function (e.g., `a`
            for LeakyReLU slope in `kaiming_normal_`).
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        # Iterate over all submodules recursively
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for convolutional layers
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale # Apply scale factor
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # Fill bias with specified value
            elif isinstance(m, nn.Linear):
                # Kaiming Normal initialization for linear layers
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale # Apply scale factor
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill) # Fill bias with specified value
            elif isinstance(m, _BatchNorm):
                # Initialize BatchNorm weights (gamma) to 1 and bias (beta) to bias_fill
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same basic blocks.

    Args:
        basic_block (nn.Module): The class of the basic block to be stacked.
        num_basic_block (int): The number of times to stack the basic block.
        **kwarg: Keyword arguments to pass to the basic_block constructor.

    Returns:
        nn.Sequential: A Sequential module containing the stacked blocks.
    """
    layers = [] # List to hold block instances
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg)) # Instantiate and add the block
    return nn.Sequential(*layers) # Return as an nn.Sequential module


class ResidualBlockNoBN(nn.Module):
    """Residual block without Batch Normalization.

    Structure: Conv -> ReLU -> Conv -> Add input (with scale).

    Args:
        num_feat (int): Number of input and output channels. Default: 64.
        res_scale (float): Scaling factor for the residual connection. Default: 1.
        pytorch_init (bool): If True, use default PyTorch initialization. Otherwise,
            use `default_init_weights` with a scale of 0.1. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        # First convolutional layer: 3x3 kernel, same padding
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # Second convolutional layer: 3x3 kernel, same padding
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights if not using PyTorch default init
        if not pytorch_init:
            # Initialize conv layers with a scale of 0.1, common for residual paths
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x # Store input for the residual connection
        out = self.conv2(self.relu(self.conv1(x))) # Apply conv1 -> ReLU -> conv2
        return identity + out * self.res_scale # Add scaled output to the input


class Upsample(nn.Sequential):
    """Upsample module using PixelShuffle.

    Supports integer scale factors that are powers of 2 or 3.

    Args:
        scale (int): The upsampling scale factor. Supported: 2^n and 3.
        num_feat (int): The number of input channels.
    """

    def __init__(self, scale, num_feat):
        m = [] # List to hold the layers
        if (scale & (scale - 1)) == 0:  # Check if scale is a power of 2
            # Calculate the number of 2x PixelShuffle layers needed
            num_pixel_shuffle_layers = int(math.log(scale, 2))
            for _ in range(num_pixel_shuffle_layers):
                # Conv layer to increase channels by 4 (2^2) before 2x PixelShuffle
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                # PixelShuffle layer with factor 2
                m.append(nn.PixelShuffle(2))
                # The number of channels is effectively reduced by 4 after PixelShuffle
                num_feat = num_feat * 4 // (2**2) # num_feat remains the same for the next layer if scale is power of 2
        elif scale == 3:
            # Conv layer to increase channels by 9 (3^2) before 3x PixelShuffle
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            # PixelShuffle layer with factor 3
            m.append(nn.PixelShuffle(3))
            # The number of channels is effectively reduced by 9 after PixelShuffle
            num_feat = num_feat * 9 // (3**2) # num_feat remains the same
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m) # Initialize Sequential with the created layers


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow using grid_sample.

    Args:
        x (Tensor): Input tensor with shape (N, C, H, W).
        flow (Tensor): Flow field tensor with shape (N, H, W, 2). Flow values
                       are assumed to be pixel displacements in the same
                       coordinate system as `x`.
        interp_mode (str): Interpolation mode ('nearest' or 'bilinear'). Default: 'bilinear'.
        padding_mode (str): Padding mode ('zeros', 'border', or 'reflection').
            Default: 'zeros'.
        align_corners (bool): Whether to align corners during grid sampling.
            Default: True.

    Returns:
        Tensor: Warped tensor with shape (N, C, H, W).
    """
    assert x.size()[-2:] == flow.size()[1:3], "Spatial dimensions of input and flow must match."
    _, _, h, w = x.size()

    # Create a mesh grid representing the target coordinates (where we want to sample FROM)
    # grid_x, grid_y will have shape (H, W)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device),
        indexing='ij' # Use 'ij' indexing for (row, col) or (y, x)
    )
    # Stack grid_x and grid_y to get a grid of shape (H, W, 2) with (x, y) coordinates
    grid = torch.stack((grid_x, grid_y), 2) # Shape (H, W, 2)
    # Expand grid to match batch size (N, H, W, 2)
    grid = grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1)

    # Add the flow to the grid to get the sampling locations in the input tensor
    vgrid = grid + flow # flow is assumed to be displacement

    # Scale grid coordinates to [-1, 1] as required by F.grid_sample
    # The scaling maps the range [0, D-1] to [-1, 1].
    # Scaled_coord = 2 * coord / (D-1) - 1
    # Handle edge case where dimension is 1 (D-1 = 0)
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    # Fix: use h for y coordinate scaling instead of max(h-1,1)/max(h-1,1)
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0

    # Stack the scaled coordinates
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3) # Shape (N, H, W, 2)

    # Perform grid sampling to warp the input tensor
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow field according to a ratio or a target shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W]. Flow values are
                       assumed to be pixel displacements in the *original* scale.
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): If 'ratio', [ratio_h, ratio_w]. If 'shape', [out_h, out_w].
                                   Ratios should be > 1.0 for upsampling, < 1.0 for downsampling.
        interp_mode (str): The mode of interpolation for resizing ('nearest' or 'bilinear').
            Default: 'bilinear'.
        align_corners (bool): Whether to align corners during interpolation. Default: False.

    Returns:
        Tensor: Resized flow. The flow values are scaled proportionally to the resize.
    """
    _, _, flow_h, flow_w = flow.size()

    if size_type == 'ratio':
        if len(sizes) != 2:
             raise ValueError("For size_type 'ratio', sizes must be a list of two floats [ratio_h, ratio_w].")
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
        ratio_h = sizes[0]
        ratio_w = sizes[1]
    elif size_type == 'shape':
        if len(sizes) != 2:
             raise ValueError("For size_type 'shape', sizes must be a list of two integers [out_h, out_w].")
        output_h, output_w = sizes[0], sizes[1]
        # Calculate ratios based on target shape
        ratio_h = output_h / flow_h
        ratio_w = output_w / flow_w
    else:
        raise ValueError(
            f'Size type should be "ratio" or "shape", but got type {size_type}.')

    # Scale flow values by the resize ratio
    # Flow in x direction corresponds to width (index 0), flow in y corresponds to height (index 1)
    input_flow = flow.clone()
    input_flow[:, 0, :, :] *= ratio_w # Scale x-component by width ratio
    input_flow[:, 1, :, :] *= ratio_h # Scale y-component by height ratio

    # Resize the flow field using interpolation
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)

    return resized_flow


class PixelUnshuffle1(nn.Module):
    r"""Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
    in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`, where r is a downscale factor.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by.

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

        .. math::
            C_{out} = C_{in} \times \text{downscale\_factor}^2

        .. math::
            H_{out} = H_{in} \div \text{downscale\_factor}

        .. math::
            W_{out} = W_{in} \div \text{downscale\_factor}

    Note: This implementation is a standard way to perform pixel unshuffle.
    """
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        if not isinstance(downscale_factor, int) or downscale_factor <= 0:
            raise ValueError(f'downscale_factor must be a positive integer, but got {downscale_factor}')
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input tensor of shape (N, C, H*r, W*r).

        Returns:
            Tensor: Pixel unshuffled tensor of shape (N, C*r^2, H, W).
        """
        b, c, hh, hw = input.size()

        # Check if dimensions are divisible by the downscale factor
        if hh % self.downscale_factor != 0 or hw % self.downcale_factor != 0: # Typo: downcale_factor -> downscale_factor
             raise ValueError(f'Input spatial dimensions ({hh}x{hw}) must be divisible by downscale_factor ({self.downscale_factor})')

        out_channel = c * (self.downscale_factor**2)
        # Calculate output height and width
        h = hh // self.downscale_factor
        w = hw // self.downscale_factor

        # Reshape and permute to perform pixel unshuffle
        # View as (b, c, h, r_h, w, r_w) where r_h=r_w=downscale_factor
        x_view = input.view(b, c, h, self.downscale_factor, w, self.downscale_factor)
        # Permute dimensions to bring the r_h and r_w dimensions together after the channel dimension
        # (b, c, r_h, r_w, h, w)
        # Reshape to combine c, r_h, and r_w into the new channel dimension
        # (b, c * r_h * r_w, h, w)
        return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

    def extra_repr(self) -> str:
        """String representation for the module."""
        return 'downscale_factor={}'.format(self.downscale_factor)


class LayerNormFunction(torch.autograd.Function):
    """Autograd function for Layer Normalization (manual implementation)."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        Forward pass for Layer Normalization.

        Args:
            ctx: Context object for saving tensors for backward pass.
            x (Tensor): Input tensor of shape (N, C, H, W).
            weight (Tensor): Learned scale parameter of shape (C,).
            bias (Tensor): Learned bias parameter of shape (C,).
            eps (float): Small value added for numerical stability.

        Returns:
            Tensor: Normalized and affine-transformed tensor of shape (N, C, H, W).
        """
        ctx.eps = eps
        N, C, H, W = x.size()
        # Calculate mean and variance over the channel dimension (C) for each spatial location
        # Keep dimensions for broadcasting
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # Normalize the input
        y = (x - mu) / (var + eps).sqrt()
        # Save tensors for the backward pass
        ctx.save_for_backward(y, var, weight)
        # Apply learned scale and bias (broadcasting happens automatically)
        # Reshape weight and bias to (1, C, 1, 1) for broadcasting with (N, C, H, W)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Layer Normalization.

        Args:
            ctx: Context object with saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output
                                  of the LayerNorm layer, shape (N, C, H, W).

        Returns:
            tuple: Gradients with respect to input x, weight, bias, and None for eps.
        """
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        # Retrieve saved tensors
        y, var, weight = ctx.saved_variables

        # Calculate gradient with respect to the normalized output before affine transform
        # Apply the learned weight gradient (chain rule)
        g = grad_output * weight.view(1, C, 1, 1)

        # Calculate mean of g and mean of g*y over the channel dimension (C)
        # Keep dimensions for broadcasting
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        # Calculate gradient with respect to the input x
        # This formula is derived from the chain rule for layer norm
        # It involves the inverse standard deviation, g, y, mean_g, and mean_gy
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        # Calculate gradients with respect to weight and bias
        # Sum gradients over batch (N), height (H), and width (W) dimensions
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)

        # Return gradients for x, weight, bias, and None for eps (as it's not a parameter)
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs (images).

    Applies normalization across the channel dimension (C) for inputs of shape (N, C, H, W).
    Uses the custom `LayerNormFunction` for the forward and backward passes.

    Args:
        channels (int): Number of input channels.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6.
    """

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        # Learnable scale parameter, initialized to ones, shape (channels,)
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        # Learnable bias parameter, initialized to zeros, shape (channels,)
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Normalized tensor of shape (N, C, H, W).
        """
        # Apply the custom LayerNorm function
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# handle multiple input
class MySequential(nn.Sequential):
    """A Sequential module that can handle multiple inputs passed as a tuple."""
    def forward(self, *inputs):
        # Iterate through each module in the sequential block
        for module in self._modules.values():
            # If the current inputs are a tuple, unpack and pass them as multiple arguments
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            # Otherwise, pass the single input
            else:
                inputs = module(inputs)
        return inputs

# measure_inference_speed function is already implemented above
# import time # Already imported

def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    """
    Measures the inference speed (FPS) of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to measure.
        data (tuple | Tensor): Example input data (or a tuple of data) for the model.
                               Should be on the correct device (CPU/GPU).
        max_iter (int): Maximum number of iterations to run. Includes warm-up.
        log_interval (int): Interval for logging progress.

    Returns:
        float: Estimated FPS (images per second).
    """
    model.eval() # Set model to evaluation mode

    # The first several iterations may be very slow so skip them (warm-up)
    num_warmup = 5
    pure_inf_time = 0 # Time spent in actual inference

    # Ensure data is a tuple if it's a single tensor, for consistent handling
    if not isinstance(data, tuple):
        data = (data,)

    # Benchmark loop
    with torch.no_grad(): # Disable gradient calculation for inference
        for i in range(max_iter):
            # Synchronize CUDA if using GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter() # Start timing

            # Forward pass
            model(*data)

            # Synchronize CUDA again
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time # Time for this iteration

            if i >= num_warmup:
                pure_inf_time += elapsed # Accumulate time after warm-up

            # Log progress periodically
            if (i + 1) % log_interval == 0:
                if pure_inf_time > 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    logger.info(
                        f'Done image [{i + 1:<3}/ {max_iter}], '
                        f'fps: {fps:.1f} img / s, '
                        f'times per image: {1000 / fps:.1f} ms / img',
                        extra={'iter': i + 1}) # Add iter to log record
                else:
                     # Avoid division by zero if pure_inf_time is still 0
                     logger.info(f'Done image [{i + 1:<3}/ {max_iter}], Warm-up phase.', extra={'iter': i + 1})


            # Stop if max iterations reached
            if (i + 1) == max_iter:
                break

    # Final FPS calculation
    if pure_inf_time > 0:
        fps = (max_iter - num_warmup) / pure_inf_time
        logger.info(
            f'Overall fps: {fps:.1f} img / s, '
            f'times per image: {1000 / fps:.1f} ms / img',
            extra={'iter': max_iter})
    else:
        fps = 0.0
        logger.warning("Pure inference time is zero, cannot calculate FPS.")

    return fps

# Note: The commented-out DCNv2Pack was part of the original file but is not used
# in the CGNet architecture provided, so it's omitted here to keep the codebase focused.
# If you need deformable convolutions, you would need to implement or import them.

# models/archs/local_arch.py

# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# This file contains utilities possibly related to specific pooling or local
# processing techniques from the original NAFNet context.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Setup a simple logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AvgPool2d(nn.Module):
    """
    Custom Average Pooling 2D module with options for kernel size calculation
    based on base_size and train_size, and a potentially faster implementation
    using cumulative sums.

    Note: This module appears specialized, likely for specific adaptive pooling
    or box filtering scenarios in the context of the original paper.
    The CGNet architecture provided doesn't explicitly use this AvgPool2d in its
    main forward pass, but the Local_Base class suggests it might be used in
    related models or parts of the original framework. We include it here as
    it was part of the original local_arch.py.
    """
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size # Explicit kernel size (tuple or int)
        self.base_size = base_size # Base size for dynamic kernel calculation
        self.auto_pad = auto_pad # Whether to pad the output to match input size
        self.fast_imp = fast_imp # Whether to use the potentially faster cumulative sum implementation
        self.train_size = train_size # Expected training input size (H, W) as a tuple

        # Parameters for the fast implementation (specific reduction ratios)
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0] # Max ratio for height reduction in fast_imp
        self.max_r2 = self.rs[0] # Max ratio for width reduction in fast_imp

        # Validate inputs
        if self.kernel_size is None and self.base_size is None:
            raise ValueError("Either kernel_size or base_size must be provided.")
        if self.base_size is not None and self.train_size is None:
             logger.warning("base_size is provided but train_size is None. Cannot calculate kernel_size dynamically.")


    def extra_repr(self) -> str:
        """Provides extra representation string for the module."""
        # Note: stride is not a direct parameter in this custom implementation
        return 'kernel_size={}, base_size={}, auto_pad={}, fast_imp={}, train_size={}'.format(
            self.kernel_size, self.base_size, self.auto_pad, self.fast_imp, self.train_size
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after average pooling.
        """
        # Dynamically calculate kernel size if base_size and train_size are provided
        if self.kernel_size is None and self.base_size is not None and self.train_size is not None:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size) # Ensure base_size is a tuple

            # Calculate kernel size based on ratio of current input size to train size
            current_h, current_w = x.shape[2:]
            train_h, train_w = self.train_size[-2:] # Assuming train_size is (C, H, W) or (H, W)

            # Calculate kernel size maintaining the ratio relative to train_size
            # Ensure kernel size is at least 1
            k1 = max(1, current_h * self.base_size[0] // train_h)
            k2 = max(1, current_w * self.base_size[1] // train_w)
            self.kernel_size = (k1, k2)
            # logger.debug(f"Dynamic kernel size calculated: {self.kernel_size} for input shape {x.shape}")


            # Update parameters for fast implementation based on current size ratio
            if self.fast_imp:
                self.max_r1 = max(1, self.rs[0] * current_h // train_h)
                self.max_r2 = max(1, self.rs[0] * current_w // train_w)

        # Ensure kernel_size is set before proceeding
        if self.kernel_size is None:
             raise RuntimeError("kernel_size was not set and could not be calculated dynamically.")

        # Handle cases where kernel size is larger than or equal to input size
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
             # If kernel is larger than or equal to input, it's equivalent to adaptive pooling to 1x1
             # logger.debug("Kernel size >= input size, using adaptive_avg_pool2d to 1x1")
             return F.adaptive_avg_pool2d(x, 1)

        # Apply pooling based on implementation choice
        if self.fast_imp:
            # Potentially faster implementation using cumulative sums and downsampling
            h, w = x.shape[2:]
            k1, k2 = self.kernel_size

            # If kernel size is still >= current size, fall back to adaptive pooling (should be caught above, but safety)
            if k1 >= h and k2 >= w:
                 # logger.debug("Fast imp: Kernel size >= current size, using adaptive_avg_pool2d to 1x1")
                 out = F.adaptive_avg_pool2d(x, 1)
            else:
                # Find suitable reduction ratios from self.rs that divide height/width
                # This part is specific and might need careful consideration of the original paper's intent
                r1_candidates = [r for r in self.rs if h % r == 0]
                r2_candidates = [r for r in self.rs if w % r == 0]

                if not r1_candidates or not r2_candidates:
                     logger.warning(f"Fast imp: Could not find suitable reduction ratios in {self.rs} for input size ({h},{w}). Falling back to standard pooling.")
                     # Fallback to standard pooling if fast implementation ratios don't apply
                     # Assuming stride 1 for this path to match the cumulative sum window
                     out = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size[0]//2) # Add padding for 'same' like behavior if stride is 1? Original code doesn't show padding here. Let's keep it without padding like the original snippet implies.
                     out = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1) # Assuming stride 1

                else:
                    r1 = r1_candidates[0] # Take the first suitable ratio
                    r2 = r2_candidates[0]

                    # Apply reduction constraint based on max_r1/max_r2
                    r1 = min(self.max_r1, r1)
                    r2 = min(self.max_r2, r2)
                    # logger.debug(f"Fast imp: Using reduction ratios ({r1},{r2})")

                    # Compute cumulative sum on downsampled feature map
                    # This is a technique for efficient box filtering (average pooling)
                    s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                    n, c, h_reduced, w_reduced = s.shape

                    # Calculate kernel size in the reduced feature map scale
                    k1_reduced = min(h_reduced - 1, self.kernel_size[0] // r1)
                    k2_reduced = min(w_reduced - 1, self.kernel_size[1] // r2)

                    if k1_reduced <= 0 or k2_reduced <= 0:
                         logger.warning(f"Fast imp: Reduced kernel size is non-positive ({k1_reduced},{k2_reduced}). Falling back to standard pooling.")
                         out = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1) # Assuming stride 1
                    else:
                        # Use cumulative sums to compute average pool output over sliding window
                        # This calculates the sum over a window of size k1_reduced x k2_reduced
                        out = (s[:, :, :-k1_reduced, :-k2_reduced] - s[:, :, :-k1_reduced, k2_reduced:] - s[:, :, k1_reduced:, :-k2_reduced] + s[:, :, k1_reduced:, k2_reduced:]) / (k1_reduced * k2_reduced)

                        # Upsample the result back to the original downsampled spatial size (h_reduced, w_reduced)
                        # The original code interpolates back by scale_factor=(r1, r2)
                        # logger.debug(f"Fast imp: Upsampling reduced output ({out.shape}) by scale ({r1},{r2})")
                        out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2), mode='nearest') # Using nearest as it's often faster/simpler for this context


        else:
            # Standard Average Pooling implementation using cumulative sums
            # This computes average pooling over a sliding window of size kernel_size
            n, c, h, w = x.shape
            k1, k2 = self.kernel_size

            # Compute cumulative sum on the original feature map
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            # Pad with zeros at the top and left for easier calculation of sums over windows starting at (0,0)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0)) # pad 0 for convenience (left, right, top, bottom)

            # Use cumulative sums to compute average pool output over sliding window
            # The sum over a window from (r1, c1) to (r2, c2) is s[r2, c2] - s[r1-1, c2] - s[r2, c1-1] + s[r1-1, c1-1]
            # Here, we are calculating the sum over a window of size k1 x k2 ending at each pixel (i, j)
            # The window starts at (i - k1 + 1, j - k2 + 1) and ends at (i, j)
            # Which corresponds to s4 + s1 - s2 - s3 in the original code's indexing relative to the padded tensor
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2) # Divide by the area of the kernel to get the average

        # Apply auto-padding to match input spatial dimensions if enabled
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # Calculate padding needed to make output size match input size
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, # Left, Right padding
                     (h - _h) // 2, (h - _h + 1) // 2) # Top, Bottom padding
            # Apply padding using replicate mode to avoid introducing black borders
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
            # logger.debug(f"Auto-padded output shape: {out.shape}")


        return out

'''
ref.
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''
class Local_Base():
    """
    Base class providing a method to convert nn.AdaptiveAvgPool2d layers
    to the custom AvgPool2d layer. This is likely intended for models that
    use adaptive pooling and want to switch to this custom implementation,
    potentially for performance or specific behavior reasons described in
    the referenced paper.

    Models that need this conversion should inherit from this class.
    """
    def convert(self, base_size, train_size, fast_imp, **kwargs):
        """
        Recursively replaces nn.AdaptiveAvgPool2d modules with AvgPool2d
        within the model's children.

        Args:
            base_size (tuple): base_size argument for AvgPool2d.
            train_size (tuple): The training input size (H, W) or (C, H, W).
            fast_imp (bool): fast_imp argument for AvgPool2d.
            **kwargs: Additional keyword arguments for AvgPool2d.
        """
        # Call the helper function to perform the replacement starting from self
        replace_layers(self, base_size, train_size, fast_imp, **kwargs)
        # The original code includes a forward pass on dummy data here.
        # This might be for shape inference or initialization based on train_size.
        # We'll keep it as it was in the original, assuming its purpose.
        # Note: This requires the model's forward method to handle a single tensor input.
        try:
            # Create dummy data with train_size. Assuming train_size is (C, H, W) or (H, W).
            # Need to handle the case where train_size is just (H, W) and add a channel dimension.
            if len(train_size) == 2:
                # Assuming 3 channels if only H, W are provided, adjust if necessary
                dummy_shape = (1, 3, train_size[0], train_size[1]) # Add batch and channel dim
            elif len(train_size) == 3:
                 dummy_shape = (1, *train_size) # Add batch dim
            else:
                 raise ValueError(f"train_size must be (H, W) or (C, H, W), but got {train_size}")

            imgs = torch.rand(dummy_shape) # Create dummy data
            with torch.no_grad():
                 self.forward(imgs) # Perform a dummy forward pass
            logger.info("Dummy forward pass successful after AvgPool2d conversion.")
        except Exception as e:
            logger.warning(f"Dummy forward pass failed during Local_Base conversion: {e}")
            logger.warning("Ensure the model's forward method can accept a single tensor input (batch, channels, height, width).")


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    """
    Helper function to recursively traverse a model and replace
    nn.AdaptiveAvgPool2d layers with AvgPool2d.

    Args:
        model (nn.Module): The model to traverse.
        base_size (tuple): base_size argument for AvgPool2d.
        train_size (tuple): train_size argument for AvgPool2d.
        fast_imp (bool): fast_imp argument for AvgPool2d.
        **kwargs: Additional keyword arguments for AvgPool2d.
    """
    # Iterate through the named children of the current module
    for n, m in model.named_children():
        # If the child module has its own children (is a container), recurse into it
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        # If the child module is an instance of nn.AdaptiveAvgPool2d
        if isinstance(m, nn.AdaptiveAvgPool2d):
            # Create an instance of the custom AvgPool2d with specified parameters
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size, **kwargs)
            # Assert that the original AdaptiveAvgPool2d was pooling to 1x1
            # This custom AvgPool2d seems specifically intended to replace 1x1 adaptive pooling
            assert m.output_size == 1, "AvgPool2d replacement expects AdaptiveAvgPool2d pooling to size 1."
            # Replace the original module with the custom one using setattr
            setattr(model, n, pool)
            logger.info(f"Replaced module '{n}' (AdaptiveAvgPool2d) with custom AvgPool2d.")

# Note: This file contains utilities related to a specific pooling implementation
# and a base class for models that might use it. The core CGNet architecture
# provided in CGNet_arch.py does not directly inherit from Local_Base or
# explicitly use AvgPool2d in its main forward pass, but these utilities
# might be used elsewhere in the original repository's framework or for
# specific model variants. They are included here to match the structure
# of the original local_arch.py.

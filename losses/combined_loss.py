# # losses/combined_loss.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # Import the SSIM function from pytorch-msssim
# from pytorch_msssim import ssim # You need to install 'pytorch-msssim'
# import logging

# # Import the new GradientLoss module
# from .gradient_loss import GradientLoss

# # Import the LossWeights module if learnable weights are used
# # Ensure learnable_weights.py exists and contains the LossWeights class
# try:
#     from .learnable_weights import LossWeights
#     _has_learnable_weights_module = True
# except ImportError:
#     _has_learnable_weights_module = False
#     # Define a dummy LossWeights if the module is not found,
#     # so the class definition doesn't fail, but learnable_weights won't work.
#     class LossWeights(nn.Module):
#         def __init__(self, num_losses=4): # Updated num_losses to 4
#             super().__init__()
#             logging.warning("LossWeights module not found. Learnable weights are disabled.")
#             # Define parameters corresponding to the expected number of losses (pixel, ssim, feature, gradient)
#             self.log_lambda_pixel = nn.Parameter(torch.zeros(1))
#             self.log_lambda_ssim = nn.Parameter(torch.zeros(1))
#             self.log_lambda_feat = nn.Parameter(torch.zeros(1))
#             self.log_lambda_grad = nn.Parameter(torch.zeros(1)) # Added gradient lambda


#         def forward(self, *losses):
#              logging.warning("LossWeights forward called but module not available. Returning sum of losses.")
#              return sum(losses)

#         def get_lambdas(self):
#              logging.warning("LossWeights get_lambdas called but module not available. Returning ones.")
#              # Return ones corresponding to the expected number of losses
#              return [torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1)] # Updated to 4 ones


# logger = logging.getLogger(__name__)
# if not logger.handlers:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# class CombinedLoss(nn.Module):
#     """
#     Combines pixel-wise, SSIM, perceptual feature, and gradient losses.
#     Can use fixed weights or learnable weights for combination.
#     Includes adjustable pixel loss weight for fixed weight mode.
#     """
#     def __init__(self, pixel_loss_type='mae', pixel_loss_weight=1.0,
#                  ssim_loss_weight=1.0, feature_loss_weight=1.0,
#                  gradient_loss_weight=1.0, # Added gradient_loss_weight
#                  learnable_weights=False):
#         """
#         Args:
#             pixel_loss_type (str): Type of pixel-wise loss ('mse' or 'mae'). Defaults to 'mae'.
#             pixel_loss_weight (float): Weight for the pixel loss component (if not learnable). Defaults to 1.0.
#             ssim_loss_weight (float): Weight for the SSIM loss component (if not learnable). Defaults to 1.0.
#             feature_loss_weight (float): Weight for the perceptual feature loss component (if not learnable). Defaults to 1.0.
#             gradient_loss_weight (float): Weight for the gradient loss component (if not learnable). Defaults to 1.0. # Added arg
#             learnable_weights (bool): Whether to use learnable weights for combining losses. Defaults to False.
#                                       Requires LossWeights module to be available.
#         """
#         super().__init__()
#         self.pixel_loss_type = pixel_loss_type
#         self.pixel_loss_weight = pixel_loss_weight # Store the weight
#         self.ssim_loss_weight = ssim_loss_weight
#         self.feature_loss_weight = feature_loss_weight
#         self.gradient_loss_weight = gradient_loss_weight # Store the weight
#         self.learnable_weights = learnable_weights and _has_learnable_weights_module # Ensure module is available if requested

#         # Define the pixel-wise loss criterion
#         if self.pixel_loss_type == 'mse':
#             self.pixel_criterion = nn.MSELoss()
#             logger.info("Using MSE for pixel-wise loss.")
#         elif self.pixel_loss_type == 'mae':
#             self.pixel_criterion = nn.L1Loss()
#             logger.info("Using MAE for pixel-wise loss.")
#         else:
#             raise ValueError(f"Unsupported pixel_loss_type: {pixel_loss_type}. Choose 'mse' or 'mae'.")

#         # SSIM loss (Note: ssim function returns similarity, we want dissimilarity/loss)
#         self.ssim_criterion = ssim # Use the ssim function from the library directly

#         # Perceptual feature loss (using L1 loss on features)
#         self.feature_criterion = nn.L1Loss()
#         logger.info("Using L1 loss for perceptual feature loss.")

#         # Define Gradient Loss
#         self.gradient_loss_fn = GradientLoss(loss_type='l1') # Using L1 for gradient difference
#         logger.info("Using Gradient Loss (L1 difference).")


#         # Initialize learnable weights module if enabled and module is available
#         if self.learnable_weights:
#             # Number of losses: pixel, ssim, feature, gradient = 4
#             self.loss_weights_module = LossWeights(num_losses=4) # Updated num_losses to 4
#             logger.info("Using learnable loss weights.")
#         else:
#             self.loss_weights_module = None # Use fixed weights
#             # Updated log message to include pixel_loss_weight and gradient_loss_weight
#             logger.info(f"Using fixed loss weights: pixel={self.pixel_loss_weight}, ssim={self.ssim_loss_weight}, feature={self.feature_loss_weight}, gradient={self.gradient_loss_weight}")


#     def forward(self, output, target, ae_features_pred, ae_features_gt):
#         """
#         Calculates the total combined loss.

#         Args:
#             output (torch.Tensor): Model output (denoised image) from CGNet. Expected range [0, 1].
#             target (torch.Tensor): Ground truth NDCT image. Expected range [0, 1].
#             ae_features_pred (torch.Tensor): Feature maps from AE encoder for the model output.
#             ae_features_gt (torch.Tensor): Feature maps from AE encoder for the ground truth.

#         Returns:
#             torch.Tensor: The total combined loss (scalar tensor).
#             dict: A dictionary containing individual loss values for logging.
#         """
#         # Calculate individual losses
#         pixel_loss = self.pixel_criterion(output, target)

#         # SSIM loss: 1 - SSIM similarity.
#         # data_range should match the range of your input tensors (output and target).
#         # Since we normalized images to [0, 1] and AE output uses Sigmoid, data_range=1.0 is appropriate.
#         ssim_val = self.ssim_criterion(output, target, data_range=1.0, size_average=True) # size_average=True returns a scalar
#         ssim_loss = 1.0 - ssim_val

#         # Perceptual feature loss (L1 distance between feature maps)
#         feature_loss = self.feature_criterion(ae_features_pred, ae_features_gt)

#         # Calculate Gradient Loss
#         gradient_loss = self.gradient_loss_fn(output, target)


#         # Combine losses using fixed or learnable weights
#         if self.learnable_weights and self.loss_weights_module is not None:
#             # Use learnable weights from the LossWeights module
#             # Pass pixel, ssim, feature, and gradient losses
#             total_loss = self.loss_weights_module(pixel_loss, ssim_loss, feature_loss, gradient_loss)
#             # You might want to log the current lambda values here or in the training loop
#             # lambdas = self.loss_weights_module.get_lambdas()
#             # logger.debug(f"Current learnable lambdas: pixel={lambdas[0].item():.4f}, ssim={lambdas[1].item():.4f}, feat={lambdas[2].item():.4f}, grad={lambdas[3].item():.4f}")

#         else:
#             # Use fixed weights from the constructor arguments
#             total_loss = self.pixel_loss_weight * pixel_loss + \
#                          self.ssim_loss_weight * ssim_loss + \
#                          self.feature_loss_weight * feature_loss + \
#                          self.gradient_loss_weight * gradient_loss # Added weighted gradient loss


#         # Return individual losses in a dictionary for logging/monitoring
#         loss_dict = {
#             'total_loss': total_loss,
#             'pixel_loss': pixel_loss, # Store the raw pixel loss value
#             'ssim_loss': ssim_loss,
#             'feature_loss': feature_loss,
#             'gradient_loss': gradient_loss, # Added gradient loss to dict
#             'ssim_val': ssim_val # Also return the SSIM similarity value itself
#         }
#         # If using learnable weights, add them to the dict
#         # if self.learnable_weights and self.loss_weights_module is not None:
#         #     lambdas = self.loss_weights_module.get_lambdas()
#         #     loss_dict['lambda_pixel'] = lambdas[0]
#         #     loss_dict['lambda_ssim'] = lambdas[1]
#         #     loss_dict['lambda_feat'] = lambdas[2]
#         #     loss_dict['lambda_grad'] = lambdas[3]


#         return total_loss, loss_dict # Return total loss and a dictionary of losses


# # Example Usage (for testing the module)
# if __name__ == '__main__':
#     # Create dummy tensors simulating model output, target, and AE features
#     # Assume batch size 4, 3 channels, 128x128 spatial size
#     dummy_output = torch.randn(4, 3, 128, 128, requires_grad=True)
#     dummy_target = torch.randn(4, 3, 128, 128)
#     # Assume AE reduces spatial size by 8x (128/8 = 16), and outputs 64 channels
#     dummy_ae_features_pred = torch.randn(4, 64, 16, 16)
#     dummy_ae_features_gt = torch.randn(4, 64, 16, 16)

#     print("--- Testing CombinedLoss with fixed weights (including adjustable pixel and gradient weights) ---")
#     # Instantiate CombinedLoss with fixed weights and adjusted pixel/gradient weights
#     fixed_loss_criterion = CombinedLoss(
#         pixel_loss_type='mae',
#         pixel_loss_weight=0.8, # Example: set pixel loss weight to 0.8
#         ssim_loss_weight=0.5,
#         feature_loss_weight=0.1,
#         gradient_loss_weight=0.2, # Example gradient weight
#         learnable_weights=False
#     )

#     # Calculate loss
#     total_loss_fixed, loss_dict_fixed = fixed_loss_criterion(dummy_output, dummy_target, dummy_ae_features_pred, dummy_ae_features_gt)

#     print(f"Calculated fixed weights loss: {total_loss_fixed.item():.4f}")
#     print("Individual losses (fixed weights):")
#     for key, val in loss_dict_fixed.items():
#         if isinstance(val, torch.Tensor) and val.numel() == 1:
#             print(f"  {key}: {val.item():.4f}")
#         else:
#              print(f"  {key}: {val}") # Print non-scalar tensors or other types


#     # Simulate backward pass (gradients should flow back to dummy_output)
#     total_loss_fixed.backward()
#     print(f"Gradient of total_loss_fixed w.r.t. dummy_output: {dummy_output.grad.norm().item():.4f}")

#     print("\n--- Testing CombinedLoss with learnable weights (if module available) ---")
#     if _has_learnable_weights_module:
#         # Instantiate CombinedLoss with learnable weights
#         learnable_loss_criterion = CombinedLoss(pixel_loss_type='mse', learnable_weights=True)

#         # Calculate loss
#         total_loss_learnable, loss_dict_learnable = learnable_loss_criterion(dummy_output, dummy_target, dummy_ae_features_pred, dummy_ae_features_gt)

#         print(f"Calculated learnable weights initial loss: {total_loss_learnable.item():.4f}")
#         print("Individual losses (learnable weights):")
#         for key, val in loss_dict_learnable.items():
#              if isinstance(val, torch.Tensor) and val.numel() == 1:
#                   print(f"  {key}: {val.item():.4f}")
#              else:
#                   print(f"  {key}: {val}") # Print non-scalar tensors or other types


#         # Access and print initial learnable weights
#         # Assuming LossWeights.get_lambdas() returns lambdas in order: pixel, ssim, feature, gradient
#         lambdas = learnable_loss_criterion.loss_weights_module.get_lambdas()
#         print(f"Initial learnable lambdas: pixel={lambdas[0].item():.4f}, ssim={lambdas[1].item():.4f}, feat={lambdas[2].item():.4f}, grad={lambdas[3].item():.4f}")
#         # Assuming LossWeights has log_lambda attributes
#         print(f"Initial log_lambdas: pixel={learnable_loss_criterion.loss_weights_module.log_lambda_pixel.item():.4f}, ssim={learnable_loss_criterion.loss_weights_module.log_lambda_ssim.item():.4f}, feat={learnable_loss_criterion.loss_weights_module.log_lambda_feat.item():.4f}, grad={learnable_loss_criterion.loss_weights_module.log_lambda_grad.item():.4f}")


#         # Simulate a backward pass and optimizer step to see weights update
#         optimizer_weights = torch.optim.Adam(learnable_loss_criterion.parameters(), lr=0.01)
#         optimizer_weights.zero_grad()
#         total_loss_learnable.backward(retain_graph=True) # retain_graph=True needed because dummy_output is used again

#         print(f"Gradient of total_loss_learnable w.r.t. dummy_output: {dummy_output.grad.norm().item():.4f}")

#         optimizer_weights.step()

#         print("\nSimulating one optimization step for learnable weights...")
#         lambdas_updated = learnable_loss_criterion.loss_weights_module.get_lambdas()
#         print(f"Learnable lambdas after one step: pixel={lambdas_updated[0].item():.4f}, ssim={lambdas_updated[1].item():.4f}, feat={lambdas_updated[2].item():.4f}, grad={lambdas_updated[3].item():.4f}")
#         print(f"Log_lambdas after one step: pixel={learnable_loss_criterion.loss_weights_module.log_lambda_pixel.item():.4f}, ssim={learnable_loss_criterion.loss_weights_module.log_lambda_ssim.item():.4f}, feat={learnable_loss_criterion.loss_weights_module.log_lambda_feat.item():.4f}, grad={learnable_loss_criterion.loss_weights_module.log_lambda_grad.item():.4f}")

#         # Calculate total loss again with updated weights (using the same dummy loss values)
#         updated_total_loss_learnable, _ = learnable_loss_criterion(dummy_output, dummy_target, dummy_ae_features_pred, dummy_ae_features_gt)
#         print(f"Total loss with updated weights (using same dummy inputs): {updated_total_loss_learnable.item():.4f}")

#     else:
#         print("LossWeights module not available. Skipping learnable weights test.")




# losses/combined_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the SSIM function from pytorch-msssim
from pytorch_msssim import ssim # You need to install 'pytorch-msssim'
import logging

# Import the GradientLoss module (assuming it exists)
try:
    from .gradient_loss import GradientLoss
    _has_gradient_loss_module = True
except ImportError:
    _has_gradient_loss_module = False
    logging.warning("GradientLoss module not found. Gradient loss is disabled.")
    # Define a dummy GradientLoss if the module is not found
    class GradientLoss(nn.Module):
        def __init__(self, loss_type='l1'):
            super().__init__()
            logging.warning("Dummy GradientLoss used. Always returns zero.")
        def forward(self, prediction, target):
            return torch.zeros(1, device=prediction.device) # Return zero tensor


# Import the LossWeights module
try:
    from .learnable_weights import LossWeights
    _has_learnable_weights_module = True
except ImportError:
    _has_learnable_weights_module = False
    logging.warning("LossWeights module not found. Learnable weights are disabled.")
    # Define a dummy LossWeights if the module is not found
    class LossWeights(nn.Module):
        def __init__(self, num_losses=3, initial_weights=None): # Added initial_weights here too for dummy
            super().__init__()
            logging.warning("Dummy LossWeights used. Always returns sum of losses.")
            # Define dummy parameters just to avoid errors in CombinedLoss init
            self.log_lambda_pixel = nn.Parameter(torch.zeros(1))
            self.log_lambda_ssim = nn.Parameter(torch.zeros(1))
            self.log_lambda_feat = nn.Parameter(torch.zeros(1))
            if num_losses == 4:
                 self.log_lambda_grad = nn.Parameter(torch.zeros(1))

        def forward(self, *losses):
             return sum(losses)

        def get_lambdas(self):
             # Return ones corresponding to the expected number of losses
             if hasattr(self, 'log_lambda_grad'):
                  return [torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1)]
             else:
                  return [torch.ones(1), torch.ones(1), torch.ones(1)]


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CombinedLoss(nn.Module):
    """
    Combines pixel-wise, SSIM, perceptual feature, and optionally gradient losses.
    Can use fixed weights or learnable weights for combination.
    Includes adjustable pixel loss weight and gradient loss weight for fixed weight mode.
    Allows setting initial values for learnable weights.
    """
    def __init__(self, pixel_loss_type='mae', pixel_loss_weight=1.0,
                 ssim_loss_weight=1.0, feature_loss_weight=1.0,
                 gradient_loss_weight=0.0,
                 learnable_weights=False, initial_learnable_weights=None): # Added initial_learnable_weights
        """
        Args:
            pixel_loss_type (str): Type of pixel-wise loss ('mse' or 'mae'). Defaults to 'mae'.
            pixel_loss_weight (float): Weight for the pixel loss component (if not learnable). Defaults to 1.0.
            ssim_loss_weight (float): Weight for the SSIM loss component (if not learnable). Defaults to 1.0.
            feature_loss_weight (float): Weight for the perceptual feature loss component (if not learnable). Defaults to 1.0.
            gradient_loss_weight (float): Weight for the gradient loss component (if not learnable). Defaults to 0.0.
            learnable_weights (bool): Whether to use learnable weights for combining losses. Defaults to False.
                                      Requires LossWeights module to be available.
            initial_learnable_weights (list or tuple, optional): Initial lambda values for learnable weights.
                                                                Order: [pixel, ssim, feature, (gradient)].
        """
        super().__init__()
        self.pixel_loss_type = pixel_loss_type
        self.pixel_loss_weight = pixel_loss_weight # Store the weight
        self.ssim_loss_weight = ssim_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.gradient_loss_weight = gradient_loss_weight # Store the weight

        # Only enable learnable weights if requested AND the module is available
        self.learnable_weights = learnable_weights and _has_learnable_weights_module
        self.initial_learnable_weights = initial_learnable_weights # Store initial weights


        # Define the pixel-wise loss criterion
        if self.pixel_loss_type == 'mse':
            self.pixel_criterion = nn.MSELoss()
            logger.info("Using MSE for pixel-wise loss.")
        elif self.pixel_loss_type == 'mae':
            self.pixel_criterion = nn.L1Loss()
            logger.info("Using MAE for pixel-wise loss.")
        else:
            raise ValueError(f"Unsupported pixel_loss_type: {pixel_loss_type}. Choose 'mse' or 'mae'.")

        # SSIM loss (Note: ssim function returns similarity, we want dissimilarity/loss)
        self.ssim_criterion = ssim # Use the ssim function from the library directly

        # Perceptual feature loss (using L1 loss on features)
        self.feature_criterion = nn.L1Loss()
        logger.info("Using L1 loss for perceptual feature loss.")

        # Define Gradient Loss if the module is available
        if _has_gradient_loss_module:
             self.gradient_loss_fn = GradientLoss(loss_type='l1') # Using L1 for gradient difference
             logger.info("Gradient Loss enabled (L1 difference).")
             self.num_losses = 4 # Pixel, SSIM, Feature, Gradient
        else:
             self.gradient_loss_fn = None # Gradient Loss is disabled
             logger.warning("Gradient Loss module not available. Gradient loss is disabled.")
             self.num_losses = 3 # Pixel, SSIM, Feature


        # Initialize learnable weights module if enabled and module is available
        if self.learnable_weights:
            # Pass the determined number of losses and initial weights to LossWeights
            self.loss_weights_module = LossWeights(num_losses=self.num_losses, initial_weights=self.initial_learnable_weights)
            logger.info(f"Using learnable loss weights for {self.num_losses} components.")
        else:
            self.loss_weights_module = None # Use fixed weights
            # Updated log message to include all fixed weights
            log_message = f"Using fixed loss weights: pixel={self.pixel_loss_weight}, ssim={self.ssim_loss_weight}, feature={self.feature_loss_weight}"
            if self.gradient_loss_fn is not None: # Only add gradient weight if gradient loss is enabled
                 log_message += f", gradient={self.gradient_loss_weight}"
            logger.info(log_message)


    def forward(self, output, target, ae_features_pred, ae_features_gt):
        """
        Calculates the total combined loss.

        Args:
            output (torch.Tensor): Model output (denoised image) from CGNet. Expected range [0, 1].
            target (torch.Tensor): Ground truth NDCT image. Expected range [0, 1].
            ae_features_pred (torch.Tensor): Feature maps from AE encoder for the model output.
            ae_features_gt (torch.Tensor): Feature maps from AE encoder for the ground truth.

        Returns:
            tuple: (total_loss, loss_dict)
                   total_loss (Tensor): The total combined loss.
                   loss_dict (dict): Dictionary of individual loss values for logging.
        """
        # Calculate individual losses
        pixel_loss = self.pixel_criterion(output, target)

        # SSIM loss: 1 - SSIM similarity.
        # data_range should match the range of your input tensors (output and target).
        # Since we normalized images to [0, 1] and AE output uses Sigmoid, data_range=1.0 is appropriate.
        ssim_val = self.ssim_criterion(output, target, data_range=1.0, size_average=True) # size_average=True returns a scalar
        ssim_loss = 1.0 - ssim_val

        # Perceptual feature loss (L1 distance between feature maps)
        feature_loss = self.feature_criterion(ae_features_pred, ae_features_gt)

        # Calculate Gradient Loss if enabled
        gradient_loss = torch.zeros(1, device=output.device) # Initialize to zero
        if self.gradient_loss_fn is not None:
             gradient_loss = self.gradient_loss_fn(output, target)


        # Prepare losses list for LossWeights or fixed combination
        losses_list = [pixel_loss, ssim_loss, feature_loss]
        if self.gradient_loss_fn is not None:
             losses_list.append(gradient_loss)


        # Combine losses using fixed or learnable weights
        if self.learnable_weights and self.loss_weights_module is not None:
            # Use learnable weights from the LossWeights module
            total_loss = self.loss_weights_module(*losses_list) # Pass the list of losses

        else:
            # Use fixed weights from the constructor arguments
            total_loss = self.pixel_loss_weight * pixel_loss + \
                         self.ssim_loss_weight * ssim_loss + \
                         self.feature_loss_weight * feature_loss
            if self.gradient_loss_fn is not None:
                 total_loss += self.gradient_loss_weight * gradient_loss # Add weighted gradient loss


        # Return individual losses in a dictionary for logging/monitoring
        loss_dict = {
            'total_loss': total_loss,
            'pixel_loss': pixel_loss, # Store the raw pixel loss value
            'ssim_loss': ssim_loss,
            'feature_loss': feature_loss,
            'ssim_val': ssim_val # Also return the SSIM similarity value itself
        }
        if self.gradient_loss_fn is not None:
             loss_dict['gradient_loss'] = gradient_loss # Add gradient loss to dict

        # If using learnable weights, add them to the dict for logging
        if self.learnable_weights and self.loss_weights_module is not None:
            lambdas = self.loss_weights_module.get_lambdas()
            # Assuming order: pixel, ssim, feature, [gradient]
            loss_dict['lambda_pixel'] = lambdas[0]
            loss_dict['lambda_ssim'] = lambdas[1]
            loss_dict['lambda_feat'] = lambdas[2]
            if self.gradient_loss_fn is not None:
                 loss_dict['lambda_grad'] = lambdas[3]


        return total_loss, loss_dict # Return total loss and a dictionary of losses


# Example Usage (for testing the module)
if __name__ == '__main__':
    # Create dummy tensors simulating model output, target, and AE features
    # Assume batch size 4, 3 channels, 128x128 spatial size
    dummy_output = torch.randn(4, 3, 128, 128, requires_grad=True)
    dummy_target = torch.randn(4, 3, 128, 128)
    # Assume AE reduces spatial size by 8x (128/8 = 16), and outputs 64 channels
    dummy_ae_features_pred = torch.randn(4, 64, 16, 16)
    dummy_ae_features_gt = torch.randn(4, 64, 16, 16)

    # Dummy GradientLoss module if not available
    if not _has_gradient_loss_module:
        class GradientLoss(nn.Module):
            def __init__(self, loss_type='l1'):
                super().__init__()
                print("Using dummy GradientLoss for testing.")
            def forward(self, prediction, target):
                return torch.zeros(1, device=prediction.device)
        _has_gradient_loss_module = True # Pretend it's available for this test block


    print("--- Testing CombinedLoss with fixed weights (including adjustable pixel and gradient weights) ---")
    # Instantiate CombinedLoss with fixed weights and adjusted pixel/gradient weights
    fixed_loss_criterion = CombinedLoss(
        pixel_loss_type='mae',
        pixel_loss_weight=0.8, # Example: set pixel loss weight to 0.8
        ssim_loss_weight=0.5,
        feature_loss_weight=0.1,
        gradient_loss_weight=0.2, # Example gradient weight
        learnable_weights=False
    )

    # Calculate loss
    total_loss_fixed, loss_dict_fixed = fixed_loss_criterion(dummy_output, dummy_target, dummy_ae_features_pred, dummy_ae_features_gt)

    print(f"Calculated fixed weights loss: {total_loss_fixed.item():.4f}")
    print("Individual losses (fixed weights):")
    for key, val in loss_dict_fixed.items():
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            print(f"  {key}: {val.item():.4f}")
        else:
             print(f"  {key}: {val}") # Print non-scalar tensors or other types


    # Simulate backward pass (gradients should flow back to dummy_output)
    total_loss_fixed.backward()
    print(f"Gradient of total_loss_fixed w.r.t. dummy_output: {dummy_output.grad.norm().item():.4f}")

    print("\n--- Testing CombinedLoss with learnable weights (if module available) ---")
    if _has_learnable_weights_module:
        # Instantiate CombinedLoss with learnable weights and custom initial weights
        initial_weights_test = [15.0, 10.0, 50.0] # Initial weights for pixel, ssim, feature
        if _has_gradient_loss_module:
             initial_weights_test.append(2.0) # Add initial gradient weight if enabled

        learnable_loss_criterion = CombinedLoss(
            pixel_loss_type='mse',
            learnable_weights=True,
            initial_learnable_weights=initial_weights_test # Pass initial weights
        )

        # Need to provide the individual raw losses to the forward pass for learnable weights
        dummy_pixel_loss_val = learnable_loss_criterion.pixel_criterion(dummy_output, dummy_target)
        dummy_ssim_loss_val = 1.0 - learnable_loss_criterion.ssim_criterion(dummy_output, dummy_target, data_range=1.0, size_average=True)
        dummy_feature_loss_val = learnable_loss_criterion.feature_criterion(dummy_ae_features_pred, dummy_ae_features_gt)
        losses_to_pass = [dummy_pixel_loss_val, dummy_ssim_loss_val, dummy_feature_loss_val]
        if _has_gradient_loss_module:
             dummy_gradient_loss_val = learnable_loss_criterion.gradient_loss_fn(dummy_output, dummy_target)
             losses_to_pass.append(dummy_gradient_loss_val)


        # Calculate loss
        total_loss_learnable, loss_dict_learnable = learnable_loss_criterion(*losses_to_pass)


        print(f"Calculated learnable weights initial loss: {total_loss_learnable.item():.4f}")
        print("Individual losses (learnable weights):")
        for key, val in loss_dict_learnable.items():
             if isinstance(val, torch.Tensor) and val.numel() == 1:
                  print(f"  {key}: {val.item():.4f}")
             else:
                  print(f"  {key}: {val}") # Print non-scalar tensors or other types


        # Access and print initial learnable weights
        lambdas = learnable_loss_criterion.loss_weights_module.get_lambdas()
        lambda_names = ['pixel', 'ssim', 'feature']
        if _has_gradient_loss_module:
             lambda_names.append('grad')
        print("Initial learnable lambdas:")
        for name, val in zip(lambda_names, lambdas):
             print(f"  lambda_{name}={val.item():.4f}")

        # Assuming LossWeights has log_lambda attributes
        print("Initial log_lambdas:")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_pixel'):
             print(f"  log_lambda_pixel={learnable_loss_criterion.loss_weights_module.log_lambda_pixel.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_ssim'):
             print(f"  log_lambda_ssim={learnable_loss_criterion.loss_weights_module.log_lambda_ssim.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_feat'):
             print(f"  log_lambda_feat={learnable_loss_criterion.loss_weights_module.log_lambda_feat.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_grad'):
             print(f"  log_lambda_grad={learnable_loss_criterion.loss_weights_module.log_lambda_grad.item():.4f}")


        # Simulate a backward pass and optimizer step to see weights update
        # Need to include the loss_weights_module parameters in the optimizer
        optimizer_weights = torch.optim.Adam(learnable_loss_criterion.parameters(), lr=0.01)
        optimizer_weights.zero_grad()
        # Need to ensure the inputs to the loss calculation have requires_grad=True
        # In the real training loop, the model output (dummy_output here) has requires_grad=True
        # The raw individual losses (pixel_loss, ssim_loss, etc.) will then have gradients
        # propagated through them.
        # To make the dummy losses require grad for this test:
        dummy_pixel_loss_val.requires_grad_(True)
        dummy_ssim_loss_val.requires_grad_(True)
        dummy_feature_loss_val.requires_grad_(True)
        if _has_gradient_loss_module:
             dummy_gradient_loss_val.requires_grad_(True)


        total_loss_learnable.backward(retain_graph=True) # retain_graph=True needed because dummy_output is used again

        # The gradients will be on the individual raw loss tensors now
        print(f"Gradient of total_loss_learnable w.r.t. dummy_pixel_loss_val: {dummy_pixel_loss_val.grad.item():.4f}")
        print(f"Gradient of total_loss_learnable w.r.t. dummy_ssim_loss_val: {dummy_ssim_loss_val.grad.item():.4f}")
        print(f"Gradient of total_loss_learnable w.r.t. dummy_feature_loss_val: {dummy_feature_loss_val.grad.item():.4f}")
        if _has_gradient_loss_module:
             print(f"Gradient of total_loss_learnable w.r.t. dummy_gradient_loss_val: {dummy_gradient_loss_val.grad.item():.4f}")


        optimizer_weights.step()

        print("\nSimulating one optimization step for learnable weights...")
        lambdas_updated = learnable_loss_criterion.loss_weights_module.get_lambdas()
        print("Learnable lambdas after one step:")
        lambda_names = ['pixel', 'ssim', 'feature']
        if _has_gradient_loss_module:
             lambda_names.append('grad')
        for name, val in zip(lambda_names, lambdas_updated):
             print(f"  lambda_{name}={val.item():.4f}")

        print("Log_lambdas after one step:")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_pixel'):
             print(f"  log_lambda_pixel={learnable_loss_criterion.loss_weights_module.log_lambda_pixel.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_ssim'):
             print(f"  log_lambda_ssim={learnable_loss_criterion.loss_weights_module.log_lambda_ssim.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_feat'):
             print(f"  log_lambda_feat={learnable_loss_criterion.loss_weights_module.log_lambda_feat.item():.4f}")
        if hasattr(learnable_loss_criterion.loss_weights_module, 'log_lambda_grad'):
             print(f"  log_lambda_grad={learnable_loss_criterion.loss_weights_module.log_lambda_grad.item():.4f}")


        # Calculate total loss again with updated weights (using the same dummy loss values)
        updated_total_loss_learnable, _ = learnable_loss_criterion(*losses_to_pass) # Pass the raw losses again
        print(f"Total loss with updated weights (using same dummy inputs): {updated_total_loss_learnable.item():.4f}")

    else:
        print("LossWeights module not available. Skipping learnable weights test.")

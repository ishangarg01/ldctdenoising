# # losses/learnable_weights.py

# import torch
# import torch.nn as nn
# import logging

# logger = logging.getLogger(__name__)
# if not logger.handlers:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# class LossWeights(nn.Module):
#     """
#     A PyTorch module that holds learnable parameters for weighting different loss components.
#     The weights are learned during training. Uses log-lambdas to ensure positive weights.
#     """
#     def __init__(self):
#         """
#         Initializes learnable log-lambda parameters for pixel, SSIM, and feature losses.
#         Initial log-lambdas are set to 0, meaning initial weights (lambdas) are exp(0) = 1.
#         """
#         super().__init__()
#         # Register parameters as nn.Parameter so they are included in model.parameters()
#         # and are updated by the optimizer.
#         self.log_lambda_pixel = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#         self.log_lambda_ssim = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#         self.log_lambda_feat = nn.Parameter(torch.tensor(0.0), requires_grad=True)

#         logger.info("Learnable LossWeights module initialized.")

#     def forward(self, l_pixel, l_ssim, l_feat):
#         """
#         Calculates the total weighted loss.

#         Args:
#             l_pixel (torch.Tensor): The calculated pixel-wise loss (scalar tensor).
#             l_ssim (torch.Tensor): The calculated SSIM-based loss (scalar tensor).
#             l_feat (torch.Tensor): The calculated perceptual feature loss (scalar tensor).

#         Returns:
#             torch.Tensor: The total combined loss (scalar tensor).
#         """
#         # Exponentiate the log-lambdas to get positive weights (lambdas)
#         lambda_pixel = torch.exp(self.log_lambda_pixel)
#         lambda_ssim = torch.exp(self.log_lambda_ssim)
#         lambda_feat = torch.exp(self.log_lambda_feat)

#         # Calculate the total loss as a weighted sum
#         total_loss = lambda_pixel * l_pixel + lambda_ssim * l_ssim + lambda_feat * l_feat

#         # Optional: Add a regularization term to the total loss to prevent
#         # the weights from becoming excessively large. This can help stabilize training.
#         # The regularization coefficient (e.g., 0.001) is a hyperparameter.
#         # total_loss += 0.001 * (lambda_pixel + lambda_ssim + lambda_feat) # Example L1 regularization on lambdas
#         # total_loss += 0.001 * (self.log_lambda_pixel**2 + self.log_lambda_ssim**2 + self.log_lambda_feat**2) # Example L2 regularization on log_lambdas

#         # Log the current lambda values periodically during training to monitor their evolution
#         # This logging should ideally happen in the training loop, not here in the forward pass
#         # logger.debug(f"Current lambdas: pixel={lambda_pixel.item():.4f}, ssim={lambda_ssim.item():.4f}, feat={lambda_feat.item():.4f}")

#         return total_loss

#     def get_lambdas(self):
#         """
#         Returns the current effective lambda values.
#         """
#         with torch.no_grad(): # Do not track gradients for this operation
#             lambda_pixel = torch.exp(self.log_lambda_pixel)
#             lambda_ssim = torch.exp(self.log_lambda_ssim)
#             lambda_feat = torch.exp(self.log_lambda_feat)
#         return lambda_pixel, lambda_ssim, lambda_feat


# # Example Usage (for testing the module)
# if __name__ == '__main__':
#     # Create an instance of the LossWeights module
#     loss_weights_module = LossWeights()
#     print("LossWeights module created.")
#     print(loss_weights_module)

#     # Create dummy loss values
#     dummy_l_pixel = torch.tensor(0.1, requires_grad=True)
#     dummy_l_ssim = torch.tensor(0.05, requires_grad=True)
#     dummy_l_feat = torch.tensor(0.02, requires_grad=True)

#     # Calculate the initial total loss
#     initial_total_loss = loss_weights_module(dummy_l_pixel, dummy_l_ssim, dummy_l_feat)
#     print(f"\nInitial dummy losses: pixel={dummy_l_pixel.item()}, ssim={dummy_l_ssim.item()}, feat={dummy_l_feat.item()}")
#     print(f"Initial log_lambdas: pixel={loss_weights_module.log_lambda_pixel.item()}, ssim={loss_weights_module.log_lambda_ssim.item()}, feat={loss_weights_module.log_lambda_feat.item()}")
#     print(f"Initial effective lambdas: {loss_weights_module.get_lambdas()}")
#     print(f"Initial total loss: {initial_total_loss.item()}")

#     # Simulate a backward pass and optimizer step to see weights update
#     # In a real scenario, this backward would be on the total loss of the main model
#     optimizer = torch.optim.Adam(loss_weights_module.parameters(), lr=0.01)
#     optimizer.zero_grad()
#     initial_total_loss.backward() # Backpropagate through the learnable weights
#     optimizer.step()

#     print("\nSimulating one optimization step for learnable weights...")
#     print(f"Log_lambdas after one step: pixel={loss_weights_module.log_lambda_pixel.item():.4f}, ssim={loss_weights_module.log_lambda_ssim.item():.4f}, feat={loss_weights_module.log_lambda_feat.item():.4f}")
#     print(f"Effective lambdas after one step: {loss_weights_module.get_lambdas()}")

#     # Calculate total loss again with updated weights
#     updated_total_loss = loss_weights_module(dummy_l_pixel, dummy_l_ssim, dummy_l_feat)
#     print(f"Total loss with updated weights: {updated_total_loss.item()}")



import torch
import torch.nn as nn
import logging
import math # Import math for log

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LossWeights(nn.Module):
    """
    A module to hold and apply learnable weights (lambdas) to different loss components.
    Uses log-transformed weights to ensure positivity.
    Allows setting custom initial lambda values.
    """
    def __init__(self, num_losses=4, initial_weights=None):
        """
        Args:
            num_losses (int): The number of individual loss components to weight.
                              (e.g., 3 for pixel, ssim, feature; 4 if adding gradient)
            initial_weights (list or tuple, optional): A list/tuple of initial lambda values
                                                       in the order [pixel, ssim, feature, (gradient)].
                                                       If None, initializes all lambdas to 1.0 (log_lambda = 0).
        """
        super(LossWeights, self).__init__()
        self.num_losses = num_losses

        if initial_weights is not None and len(initial_weights) != num_losses:
            raise ValueError(f"Length of initial_weights ({len(initial_weights)}) must match num_losses ({num_losses})")

        # Define learnable parameters for the log of the lambdas
        # Initialize with log(initial_weight) or log(1.0) = 0 if initial_weights is None
        if num_losses == 3:
            initial_pixel = initial_weights[0] if initial_weights else 1.0
            initial_ssim = initial_weights[1] if initial_weights else 1.0
            initial_feat = initial_weights[2] if initial_weights else 1.0

            self.log_lambda_pixel = nn.Parameter(torch.tensor([math.log(initial_pixel)]))
            self.log_lambda_ssim = nn.Parameter(torch.tensor([math.log(initial_ssim)]))
            self.log_lambda_feat = nn.Parameter(torch.tensor([math.log(initial_feat)]))
            logger.info(f"Initialized LossWeights for 3 losses with initial lambdas: pixel={initial_pixel}, ssim={initial_ssim}, feature={initial_feat}")

        elif num_losses == 4:
            initial_pixel = initial_weights[0] if initial_weights else 1.0
            initial_ssim = initial_weights[1] if initial_weights else 1.0
            initial_feat = initial_weights[2] if initial_weights else 1.0
            initial_grad = initial_weights[3] if initial_weights else 1.0 # Get initial gradient weight

            self.log_lambda_pixel = nn.Parameter(torch.tensor([math.log(initial_pixel)]))
            self.log_lambda_ssim = nn.Parameter(torch.tensor([math.log(initial_ssim)]))
            self.log_lambda_feat = nn.Parameter(torch.tensor([math.log(initial_feat)]))
            self.log_lambda_grad = nn.Parameter(torch.tensor([math.log(initial_grad)])) # Added gradient weight
            logger.info(f"Initialized LossWeights for 4 losses with initial lambdas: pixel={initial_pixel}, ssim={initial_ssim}, feature={initial_feat}, gradient={initial_grad}")

        else:
            raise ValueError(f"Unsupported number of losses: {num_losses}. Expected 3 or 4.")

        # You can add more parameters here if you have more loss components

    def forward(self, *losses):
        """
        Applies the learnable weights to the input loss components and returns the total weighted loss.

        Args:
            *losses (tuple of Tensors): Individual loss tensors (scalar).
                                        The order should match the order of lambda parameters.
                                        (e.g., pixel_loss, ssim_loss, feature_loss, [gradient_loss])

        Returns:
            Tensor: The total weighted loss (scalar).
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} loss tensors, but got {len(losses)}")

        # Get the actual lambda values by exponentiating the log_lambdas
        if self.num_losses == 3:
            lambda_pixel = torch.exp(self.log_lambda_pixel)
            lambda_ssim = torch.exp(self.log_lambda_ssim)
            lambda_feat = torch.exp(self.log_lambda_feat)
            # Calculate total weighted loss
            total_loss = lambda_pixel * losses[0] + lambda_ssim * losses[1] + lambda_feat * losses[2]
        elif self.num_losses == 4:
            lambda_pixel = torch.exp(self.log_lambda_pixel)
            lambda_ssim = torch.exp(self.log_lambda_ssim)
            lambda_feat = torch.exp(self.log_lambda_feat)
            lambda_grad = torch.exp(self.log_lambda_grad) # Get gradient lambda
             # Calculate total weighted loss
            total_loss = lambda_pixel * losses[0] + lambda_ssim * losses[1] + lambda_feat * losses[2] + lambda_grad * losses[3] # Include gradient loss
        else:
             # Should not reach here due to __init__ check
             raise RuntimeError("Invalid number of losses configured in LossWeights.")


        return total_loss

    def get_lambdas(self):
        """
        Returns the current learnable lambda values.
        """
        if self.num_losses == 3:
            return [torch.exp(self.log_lambda_pixel),
                    torch.exp(self.log_lambda_ssim),
                    torch.exp(self.log_lambda_feat)]
        elif self.num_losses == 4:
             return [torch.exp(self.log_lambda_pixel),
                     torch.exp(self.log_lambda_ssim),
                     torch.exp(self.log_lambda_feat),
                     torch.exp(self.log_lambda_grad)] # Return gradient lambda
        else:
             return [] # Should not reach here


if __name__ == '__main__':
    # Example Usage
    print("Testing LossWeights module with initial weights...")

    # Create dummy loss tensors
    dummy_pixel_loss = torch.tensor(0.0058)
    dummy_ssim_loss = torch.tensor(0.0421)
    dummy_feature_loss = torch.tensor(0.0190)
    dummy_gradient_loss = torch.tensor(0.5) # Example gradient loss value

    print("\n--- Testing with 3 losses and custom initial weights ---")
    # Instantiate LossWeights for 3 losses (pixel, ssim, feature) with custom initial weights
    initial_weights_3 = [10.0, 5.0, 20.0] # Example initial weights
    loss_weights_3 = LossWeights(num_losses=3, initial_weights=initial_weights_3)
    print(f"Initial log_lambdas (3 losses): pixel={loss_weights_3.log_lambda_pixel.item():.4f}, ssim={loss_weights_3.log_lambda_ssim.item():.4f}, feat={loss_weights_3.log_lambda_feat.item():.4f}")
    lambdas_3 = loss_weights_3.get_lambdas()
    print(f"Initial lambdas (3 losses): pixel={lambdas_3[0].item():.4f}, ssim={lambdas_3[1].item():.4f}, feat={lambdas_3[2].item():.4f}")

    # Calculate weighted loss using the forward pass
    total_weighted_loss_3 = loss_weights_3(dummy_pixel_loss, dummy_ssim_loss, dummy_feature_loss)
    print(f"Total weighted loss (3 losses): {total_weighted_loss_3.item():.4f}")


    print("\n--- Testing with 4 losses and custom initial weights ---")
    # Instantiate LossWeights for 4 losses (pixel, ssim, feature, gradient) with custom initial weights
    initial_weights_4 = [15.0, 10.0, 50.0, 2.0] # Example initial weights including gradient
    loss_weights_4 = LossWeights(num_losses=4, initial_weights=initial_weights_4)
    print(f"Initial log_lambdas (4 losses): pixel={loss_weights_4.log_lambda_pixel.item():.4f}, ssim={loss_weights_4.log_lambda_ssim.item():.4f}, feat={loss_weights_4.log_lambda_feat.item():.4f}, grad={loss_weights_4.log_lambda_grad.item():.4f}")
    lambdas_4 = loss_weights_4.get_lambdas()
    print(f"Initial lambdas (4 losses): pixel={lambdas_4[0].item():.4f}, ssim={lambdas_4[1].item():.4f}, feat={lambdas_4[2].item():.4f}, grad={lambdas_4[3].item():.4f}")

    # Calculate weighted loss using the forward pass
    total_weighted_loss_4 = loss_weights_4(dummy_pixel_loss, dummy_ssim_loss, dummy_feature_loss, dummy_gradient_loss)
    print(f"Total weighted loss (4 losses): {total_weighted_loss_4.item():.4f}")

    # Simulate optimization step (requires gradients on dummy losses in a real setup)
    # For this test, we just show the parameters exist and can be accessed.
    optimizer_4 = torch.optim.Adam(loss_weights_4.parameters(), lr=0.1)
    print("\nSimulating one optimization step for 4 losses (parameters will update if gradients were available)...")
    # In a real scenario, you would call backward() on the total loss before optimizer.step()
    # total_weighted_loss_4.backward() # Uncomment in a real setup with requires_grad=True inputs
    optimizer_4.step()

    lambdas_4_updated = loss_weights_4.get_lambdas()
    print(f"Learnable lambdas after one step: pixel={lambdas_4_updated[0].item():.4f}, ssim={lambdas_4_updated[1].item():.4f}, feat={lambdas_4_updated[2].item():.4f}, grad={lambdas_4_updated[3].item():.4f}")
    print(f"Log_lambdas after one step: pixel={loss_weights_4.log_lambda_pixel.item():.4f}, ssim={loss_weights_4.log_lambda_ssim.item():.4f}, feat={loss_weights_4.log_lambda_feat.item():.4f}, grad={loss_weights_4.log_lambda_grad.item():.4f}")

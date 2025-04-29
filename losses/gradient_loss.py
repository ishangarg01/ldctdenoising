import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GradientLoss(nn.Module):
    """
    Calculates the L1 or L2 loss between the gradients of the prediction and target.
    Uses finite differences to approximate gradients in x and y directions.
    """
    def __init__(self, loss_type='l1'):
        """
        Args:
            loss_type (str): Type of loss to apply to the gradient difference. 'l1' or 'l2'.
        """
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
            logger.info("Using L1 loss for gradient difference.")
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
            logger.info("Using MSE loss for gradient difference.")
        else:
            raise ValueError("Gradient loss_type must be 'l1' or 'l2'")

    def forward(self, prediction, target):
        """
        Args:
            prediction (Tensor): The predicted image tensor (B, C, H, W).
            target (Tensor): The target image tensor (B, C, H, W).

        Returns:
            Tensor: The calculated gradient loss.
        """
        # Calculate gradients in x direction using finite differences
        # Gradient_x = pixel(i, j) - pixel(i, j-1)
        # We slice the tensor to get the differences
        gradient_x_pred = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
        gradient_x_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        # Calculate gradients in y direction using finite differences
        # Gradient_y = pixel(i, j) - pixel(i-1, j)
        gradient_y_pred = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
        gradient_y_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Calculate loss on x gradients
        loss_x = self.loss_fn(gradient_x_pred, gradient_x_target)

        # Calculate loss on y gradients
        loss_y = self.loss_fn(gradient_y_pred, gradient_y_target)

        # Total gradient loss is the sum of x and y gradient losses
        total_gradient_loss = loss_x + loss_y

        return total_gradient_loss

if __name__ == '__main__':
    # Example Usage
    print("Testing GradientLoss module...")
    # Create dummy tensors
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)

    # Instantiate GradientLoss
    grad_loss_l1 = GradientLoss(loss_type='l1')
    grad_loss_l2 = GradientLoss(loss_type='l2')

    # Calculate loss
    loss1 = grad_loss_l1(pred, target)
    loss2 = grad_loss_l2(pred, target)

    print(f"Gradient Loss (L1): {loss1.item():.4f}")
    print(f"Gradient Loss (L2): {loss2.item():.4f}")

    # Test with a simple edge
    img_edge = torch.zeros(1, 1, 10, 10)
    img_edge[:, :, 5:, :] = 1.0 # Create a horizontal edge
    img_smooth = torch.zeros(1, 1, 10, 10) # Smooth image

    grad_loss_edge_l1 = GradientLoss(loss_type='l1')
    loss_edge = grad_loss_edge_l1(img_smooth, img_edge) # Should be high
    loss_self = grad_loss_edge_l1(img_edge, img_edge) # Should be low

    print(f"Gradient Loss (L1, Smooth vs Edge): {loss_edge.item():.4f}")
    print(f"Gradient Loss (L1, Edge vs Edge): {loss_self.item():.4f}")

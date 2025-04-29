# utils/eval_utils.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
import numpy as np

# Import metrics calculation functions
from utils.metrics import calculate_psnr, calculate_ssim # Assuming these are correct
from utils.logging_utils import setup_logger # Assuming this is correct

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger(__name__) # Use the utility to setup if not already configured


def evaluate_model(model, dataloader, device, logger_eval, epoch=None, writer=None, metrics_list=["psnr", "ssim"]):
    """
    Evaluates the model's performance on a dataset using specified metrics.

    Args:
        model (nn.Module): The trained model to evaluate (e.g., CGNet).
        dataloader (DataLoader): DataLoader for the evaluation dataset (paired LDCT, NDCT).
        device (torch.device): Device to perform evaluation on.
        logger_eval (logging.Logger): Logger for evaluation output.
        epoch (int, optional): Current epoch number for logging/TensorBoard.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter.
        metrics_list (list): List of metric names to calculate (e.g., ["psnr", "ssim"]).

    Returns:
        dict: Dictionary of average metric values over the dataset.
    """
    model.eval() # Set model to evaluation mode
    total_metrics = {metric: 0.0 for metric in metrics_list}
    num_samples = 0
    start_time = time.time()

    logger_eval.info(f"Starting evaluation{f' for epoch {epoch}' if epoch is not None else ''}...")

    with torch.no_grad(): # Disable gradient calculation during evaluation
        # Dataloader yields (inputs, targets) where inputs are LDCT and targets are NDCT
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass through the model
            outputs = model(inputs) # Denoised output

            # Calculate metrics for each image in the batch
            for i in range(inputs.size(0)):
                img_output = outputs[i] # (C, H, W) tensor - Denoised
                img_target = targets[i] # (C, H, W) tensor - Ground Truth

                # Calculate specified metrics
                if "psnr" in metrics_list:
                    # PSNR expects images in [0, 255] range (after conversion from [0, 1] float)
                    psnr_val = calculate_psnr(img_output, img_target, data_range=1.0) # Pass data_range=1.0 to the metric function
                    total_metrics["psnr"] += psnr_val
                if "ssim" in metrics_list:
                    # calculate_ssim expects (N, C, H, W), so add batch dim
                    ssim_val = calculate_ssim(img_output.unsqueeze(0), img_target.unsqueeze(0), data_range=1.0) # Pass data_range=1.0
                    total_metrics["ssim"] += ssim_val

                num_samples += 1

            # Log progress if needed (optional for eval, can be verbose)
            # if (batch_idx + 1) % 10 == 0: # Log every 10 batches
            #      logger_eval.info(f"Eval Batch [{batch_idx+1}/{len(dataloader)}]")


    # Calculate average metrics
    avg_metrics = {metric: total / num_samples if num_samples > 0 else 0.0 for metric, total in total_metrics.items()}

    elapsed_time = time.time() - start_time
    logger_eval.info(f"Evaluation finished in {elapsed_time:.2f}s. Average metrics over {num_samples} samples:")
    for metric, avg_val in avg_metrics.items():
        logger_eval.info(f"  Avg {metric.upper()}: {avg_val:.4f}")
        # Log to TensorBoard if writer is provided and epoch is not None
        if writer is not None and epoch is not None:
             writer.add_scalar(f'Eval/{metric.upper()}', avg_val, epoch)

    model.train() # Set model back to training mode (important if this is called during training)
    return avg_metrics


# Example Usage (for testing the evaluation function)
if __name__ == '__main__':
    print("--- Testing eval_utils.py ---")

    # Create a dummy model (e.g., a simple conv layer)
    class DummyModel(nn.Module):
        def __init__(self, in_channels=3):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        def forward(self, x):
            # Simulate some output (e.g., input + noise, or just input)
            # Return values in [0, 1] range
            output = torch.clamp(x + torch.randn_like(x) * 0.01, 0, 1)
            return output

    dummy_model = DummyModel(in_channels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model.to(device)
    print(f"Dummy model created on device: {device}")


    # Create dummy dataset and dataloader
    # Requires dummy data structure similar to CTDenoiseDataset expects
    dummy_root = "dummy_data_split"
    dummy_train_q_dir = os.path.join(dummy_root, "train", "quarter")
    dummy_train_f_dir = os.path.join(dummy_root, "train", "full")
    dummy_test_q_dir = os.path.join(dummy_root, "test", "quarter")
    dummy_test_f_dir = os.path.join(dummy_root, "test", "full")

    os.makedirs(dummy_train_q_dir, exist_ok=True)
    os.makedirs(dummy_train_f_dir, exist_ok=True)
    os.makedirs(dummy_test_q_dir, exist_ok=True)
    os.makedirs(dummy_test_f_dir, exist_ok=True)

    # Create dummy PNG files (e.g., 10 test pairs)
    dummy_img = Image.new('RGB', (128, 128), color = 'blue')
    num_dummy_test = 10
    for i in range(num_dummy_test):
        dummy_img.save(os.path.join(dummy_test_q_dir, f"test_img_{i:03d}.png"))
        dummy_img.save(os.path.join(dummy_test_f_dir, f"test_img_{i:03d}.png"))

    print(f"Created dummy data in {dummy_root} ({num_dummy_test} test pairs)")

    try:
        # Instantiate the testing dataset
        test_dataset = CTDenoiseDataset(root=dummy_root, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0) # Use batch size > 1 for testing

        print(f"Test DataLoader created with {len(test_dataloader)} batches.")

        # Setup a dummy logger for evaluation
        eval_logger = setup_logger('dummy_eval_logger', stream=True)

        # Evaluate the dummy model
        avg_metrics = evaluate_model(dummy_model, test_dataloader, device, eval_logger, epoch=0, metrics_list=["psnr", "ssim"])

        print("\nAverage Evaluation Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation utility test: {e}")
        print("Please ensure dummy data is created correctly and dependencies are installed.")

    finally:
        # Clean up dummy data
        # import shutil # Uncomment if you want to automatically remove dummy data
        # if os.path.exists(dummy_root):
        #     shutil.rmtree(dummy_root)
        print(f"\nDummy data left in {dummy_root}. Please remove manually if needed.")
    print("--- Eval utility test complete ---")

# scripts/test.py

import argparse
import yaml
import os
import time
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import logging # Import logging

# Import custom modules
from datasets.ct_denoise_dataset import CTDenoiseDataset # Assuming this is correct
# NOTE: Replace the placeholder define_network below with the actual import
# from models.archs import define_network
from utils.logging_utils import setup_logger # Assuming this is correct
from utils.metrics import tensor2numpy # Assuming this is correct for saving images

# Setup logger for this script
logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger(__name__) # Use the utility to setup if not already configured

# Placeholder for define_network - REMOVE THIS ONCE models.archs.__init__.py IS READY
# This should be the same placeholder as in train.py
def define_network(opt):
    """Placeholder for defining the main network (CGNet)."""
    logger.info(f"Using placeholder define_network in test.py. Defining dummy network of type: {opt['type']}")
    class DummyNetwork(nn.Module):
        def __init__(self, img_channel, width, **kwargs):
            super().__init__()
            # Simple pass-through or identity-like operation for testing
            self.identity = nn.Identity()
        def forward(self, x):
            # In a real scenario, this would be model(x)
            # For placeholder, just return the input (simulate no denoising)
            logger.debug("DummyNetwork forward pass: Returning input as output.")
            return x
    return DummyNetwork(**opt.get('args', {}))
# END OF PLACEHOLDER


def load_model(checkpoint_path, model, device):
    """
    Loads model weights from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load the state dict into.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The model with loaded weights, moved to the specified device.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint not found at {checkpoint_path}. Cannot load model.")
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading model weights from {checkpoint_path}")
    try:
        # Load checkpoint to CPU first, then move to device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Load model state dict, allowing for potential strict=False if model architecture changed slightly
        model.load_state_dict(checkpoint['model_state_dict'], strict=True) # Use strict=True by default
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model weights from {checkpoint_path}: {e}")
        logger.error("Testing cannot proceed without loading model weights.")
        raise e # Re-raise the exception

    model.to(device) # Move model to the target device
    return model


def main(config_path, checkpoint_path, output_dir):
    """Main testing function for the CGNet model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the relevant sections of the config
    model_config = config['model']
    dataset_config = config['dataset']
    # Test script doesn't need loss or optimizer configs
    # training_config = config['training'] # Not needed for inference


    # --- Setup ---
    # Setup logging - use the utility function
    # Log to console and optionally a file if output_dir is specified
    log_file = os.path.join(output_dir, 'test.log') if output_dir else None
    logger = setup_logger('test_logger', log_file=log_file, level=logging.INFO)
    logger.info(f"Starting CGNet testing using checkpoint: {checkpoint_path}")
    logger.info(f"Config:\n{yaml.dump(config, indent=4)}") # Log the full config

    # Device setup
    # Use device from config or default to cuda if available, else cpu
    device = torch.device(config.get('training', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory for saving results
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving denoised images to: {output_dir}")


    # --- Data Loading ---
    # Load test dataset and dataloader
    # We need both LDCT (input) and NDCT (for filenames/structure, though not used for inference itself)
    test_dataset = CTDenoiseDataset(
        root=dataset_config['args']['root'],
        mode='test', # Test on the test set
        transform=None # Use default ToTensor transform
    )

    if len(test_dataset) == 0:
        logger.error("No data found in the test set for inference. Please check the dataset path and preprocessing.")
        return

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=dataset_config.get('test_batch_size', 1), # Typically batch size 1 for inference/evaluation
        shuffle=False, # No need to shuffle test data
        num_workers=dataset_config.get('num_workers', 4),
        pin_memory=True and device.type == 'cuda'
    )
    logger.info(f"Test DataLoader created with {len(test_dataloader)} batches.")


    # --- Model Setup ---
    # Define the main network (CGNet)
    # NOTE: Ensure models.archs.__init__.py and the CGNet architecture file (CGNet_arch.py) are ready
    try:
         from models.archs import define_network as actual_define_network
         model = actual_define_network(model_config) # Define model (weights not loaded yet)
         logger.info("Successfully imported and defined CGNet model structure.")
    except ImportError:
         logger.error("Could not import actual define_network. Using placeholder.")
         model = define_network(model_config)
    except Exception as e:
         logger.error(f"Error defining CGNet model structure: {e}. Using placeholder.")
         model = define_network(model_config)


    # Load the trained weights from the checkpoint and move model to device
    try:
        model = load_model(checkpoint_path, model, device)
    except FileNotFoundError:
        logger.error("Exiting due to missing model checkpoint.")
        return
    except Exception as e:
        logger.error(f"Exiting due to error loading model: {e}")
        return


    # --- Inference ---
    model.eval() # Set model to evaluation mode
    start_time = time.time()

    logger.info("Starting inference on the test set...")

    # Use tqdm for progress bar in console if available (optional)
    # try:
    #     from tqdm import tqdm
    #     dataloader_iter = tqdm(test_dataloader, desc="Inference")
    # except ImportError:
    #     dataloader_iter = test_dataloader
    dataloader_iter = test_dataloader # Use standard iterator


    with torch.no_grad(): # Disable gradient calculation during inference
        # Dataloader yields (inputs, targets) where inputs are LDCT and targets are NDCT
        # We only need inputs (LDCT) for inference, but targets are included by the dataset
        for batch_idx, (inputs, targets) in enumerate(dataloader_iter):
            inputs = inputs.to(device)
            # targets = targets.to(device) # Not needed for inference, but keep for filename

            # Forward pass through the model to get denoised output
            outputs = model(inputs) # Denoised output

            # Process each image in the batch
            for i in range(inputs.size(0)):
                img_output = outputs[i] # (C, H, W) tensor - Denoised output for this image
                # Get the original filename from the target batch (assuming order is preserved)
                # CTDenoiseDataset returns (quarter_img, full_img) and filenames are derived from these
                # The dataset's __getitem__ returns (quarter_img, full_img)
                # We need the filename of the quarter_img (input) or full_img (target)
                # Let's use the filename of the quarter_img input for consistency
                # CTDenoiseDataset doesn't return filename directly in __getitem__
                # We need to modify CTDenoiseDataset to return filename OR derive it here.
                # Let's modify CTDenoiseDataset to return filename in __getitem__
                # Assuming CTDenoiseDataset is modified to return (quarter_img, full_img, filename)
                # If not modified, we'd need to reconstruct filename from index or path list.
                # Let's assume CTDenoiseDataset is updated to return filename.
                # For now, we'll use a placeholder filename derivation.
                # Placeholder filename derivation:
                # filename = f"denoised_{batch_idx * test_dataloader.batch_size + i:05d}.png"
                # Actual filename from dataset (assuming dataset returns it):
                # filename = filenames[i] # This requires modifying CTDenoiseDataset

                # TEMPORARY: Get filename from the original dataset list based on index
                # This requires access to the dataset's file list, which is not ideal
                # Let's assume CTDenoiseDataset's __getitem__ is updated to return filename.
                # If not, you'd need to pass filenames separately or modify the dataset.
                # For now, let's use a dummy filename or require dataset modification.
                # Let's assume dataset modification is done and it returns (input, target, filename)
                # The loop should be: for batch_idx, (inputs, targets, filenames) in enumerate(dataloader_iter):
                # Let's update the loop signature assuming this modification.

                # Assuming the dataloader yields (inputs, targets, filenames)
                # If not, the following line will cause an error.
                # If CTDenoiseDataset is NOT updated, you might need to get filenames from test_dataset.quarter_images[batch_idx * batch_size + i]
                # For now, let's assume the dataset returns filenames.

                # --- Assuming CTDenoiseDataset returns (inputs, targets, filenames) ---
                # If your CTDenoiseDataset is NOT updated, comment out the line below
                # and uncomment the filename derivation based on index/path list.
                # filename = filenames[i] # Get filename from the batch

                # --- If CTDenoiseDataset is NOT updated to return filenames ---
                # You need to access the original file list. This is less clean.
                # Assuming test_dataset.quarter_images contains the paths in sorted order
                original_filepath = test_dataset.quarter_images[batch_idx * test_dataloader.batch_size + i]
                filename = os.path.basename(original_filepath)
                # Replace .png with _denoised.png or similar if desired
                filename = filename.replace('.png', '_denoised.png')
                # --- End of If CTDenoiseDataset is NOT updated ---


                # Convert tensor to numpy uint8 [0, 255] for saving
                # tensor2numpy assumes input is [0, 1] float and converts to [0, 255] uint8
                img_output_np = tensor2numpy(img_output, rgb_range=1.0)

                # Save the denoised image (e.g., as PNG)
                save_path = os.path.join(output_dir, filename)
                try:
                    Image.fromarray(img_output_np).save(save_path)
                    # logger.debug(f"Saved denoised image: {save_path}") # Can be too verbose
                except Exception as e:
                    logger.error(f"Error saving denoised image {filename}: {e}")


    elapsed_time = time.time() - start_time
    logger.info(f"Inference finished in {elapsed_time:.2f}s.")
    logger.info(f"Denoised images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the trained CGNet denoiser.')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file (YAML).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained CGNet model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default='results/denoised_images',
                        help='Directory to save the denoised output images.')

    args = parser.parse_args()

    main(args.config, args.checkpoint, args.output_dir)

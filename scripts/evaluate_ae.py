# scripts/evaluate_ae.py

import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image # Import PIL for saving images
import logging # Import logging

# Import custom modules
from datasets.ct_denoise_dataset import CTDenoiseDataset # Assuming this is correct
# NOTE: Replace the placeholder define_autoencoder below with the actual import
# from models.autoencoder import define_autoencoder
from utils.logging_utils import setup_logger # Assuming this is correct
from utils.metrics import calculate_psnr, calculate_ssim, tensor2numpy # Assuming these are correct

# Setup logger for this script
logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger(__name__) # Use the utility to setup if not already configured

# Placeholder for define_autoencoder - REMOVE THIS ONCE models.autoencoder.__init__.py IS READY
# This should be the same placeholder as in train_ae.py
def define_autoencoder(opt):
    """
    Placeholder function to define AE.
    This should be replaced with the actual import from models.autoencoder.__init__.py.
    It currently returns a dummy AE model for testing purposes.
    """
    logger.info(f"Using placeholder define_autoencoder. Defining dummy AE of type: {opt['arch']['type']} with args: {opt['arch'].get('args', {})}")
    class DummyAE(nn.Module):
        def __init__(self, in_channels, base_channels, num_encoder_layers):
            super().__init__()
            layers = []
            current_channels = in_channels
            for i in range(num_encoder_layers):
                 out_channels = base_channels * (2 ** i)
                 layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
                 layers.append(nn.ReLU(inplace=True))
                 current_channels = out_channels
            self.encoder = nn.Sequential(*layers)

            decoder_layers = []
            current_channels = base_channels * (2 ** (num_encoder_layers - 1)) if num_encoder_layers > 0 else in_channels
            for i in range(num_encoder_layers - 1, -1, -1):
                 out_channels = base_channels * (2 ** (i - 1)) if i > 0 else in_channels
                 decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
                 decoder_layers.append(nn.ReLU(inplace=True) if i > 0 else nn.Identity())
                 current_channels = out_channels

            decoder_layers.append(nn.Conv2d(current_channels, in_channels, kernel_size=3, stride=1, padding=1))
            self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))
        def get_features(self, x):
             return self.encoder(x)

    ae_arch_args = opt.get('arch', {}).get('args', {})
    return DummyAE(
        in_channels=ae_arch_args.get('in_channels', 3),
        base_channels=ae_arch_args.get('base_channels', 16),
        num_encoder_layers=ae_arch_args.get('num_encoder_layers', 3)
    )
# END OF PLACEHOLDER


def main(config_path, checkpoint_path, output_dir=None, save_num_images=-1):
    """Main evaluation function for the Autoencoder."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the autoencoder and ae_evaluation sections of the config
    ae_model_config = config['autoencoder']
    ae_eval_config = config['ae_evaluation']
    dataset_config = config['dataset']

    # --- Setup ---
    # Setup logging - use the utility function
    # Log to console and optionally a file if output_dir is specified
    log_file = os.path.join(output_dir, 'evaluate_ae.log') if output_dir else None
    logger = setup_logger('evaluate_ae_logger', log_file=log_file, level=logging.INFO)
    logger.info(f"Starting AE evaluation using checkpoint: {checkpoint_path}")
    logger.info(f"Config:\n{yaml.dump(config, indent=4)}") # Log the full config

    # Device setup
    device = torch.device(ae_eval_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory for saving results if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving evaluation results to: {output_dir}")
        if save_num_images >= 0:
            logger.info(f"Will save up to {save_num_images} reconstructed images.")
        else:
             logger.info("Will save all reconstructed images.")
    else:
        logger.warning("Output directory not specified. Cannot save reconstructed images or log file.")


    # --- Data Loading ---
    # For AE evaluation, we need the full-dose (NDCT) images from the test split
    # CTDenoiseDataset loads pairs (LDCT, NDCT), so we'll use a wrapper dataset
    # to only return the target (full-dose) image.
    eval_dataset = CTDenoiseDataset(
        root=dataset_config['args']['root'],
        mode='test', # Evaluate on the test set
        transform=None # Use default ToTensor transform from CTDenoiseDataset
    )
    # Wrapper dataset to extract only the NDCT image (target) and its filename
    class NDCTOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            # CTDenoiseDataset returns (quarter_img, full_img)
            _, full_img = self.base_dataset[idx]
            # Also return the original filename for saving results
            # Assuming full_images list in base_dataset holds the paths
            full_path = self.base_dataset.full_images[idx]
            return full_img, os.path.basename(full_path) # Return NDCT image and its filename

    eval_ndct_dataset = NDCTOnlyDataset(eval_dataset)

    if len(eval_ndct_dataset) == 0:
        logger.error("No data found in the test set for evaluation. Please check the dataset path and preprocessing.")
        return

    eval_dataloader = DataLoader(
        eval_ndct_dataset,
        batch_size=dataset_config.get('test_batch_size', 1), # Use test_batch_size, typically 1 for evaluation
        shuffle=False, # No need to shuffle evaluation data
        num_workers=dataset_config.get('num_workers', 4),
        pin_memory=True and device.type == 'cuda' # Pin memory only if using CUDA
    )
    logger.info(f"Eval DataLoader created with {len(eval_dataloader)} batches.")


    # --- Model Setup ---
    # Define the Autoencoder model using the AE architecture config
    try:
         from models.autoencoder import define_autoencoder as actual_define_autoencoder
         model = actual_define_autoencoder(ae_model_config).to(device)
         logger.info("Successfully imported and defined Autoencoder model.")
    except ImportError:
         logger.error("Could not import actual define_autoencoder. Using placeholder.")
         model = define_autoencoder(ae_model_config).to(device)
    except Exception as e:
         logger.error(f"Error defining Autoencoder model: {e}. Using placeholder.")
         model = define_autoencoder(ae_model_config).to(device)

    # Load the trained weights from the checkpoint
    if not os.path.exists(checkpoint_path):
        logger.error(f"AE checkpoint not found at {checkpoint_path}. Cannot evaluate.")
        return

    logger.info(f"Loading model weights from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device) # Load directly to device
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model weights from {checkpoint_path}: {e}")
        logger.error("Evaluation cannot proceed without loading model weights.")
        return


    # --- Evaluation ---
    model.eval() # Set model to evaluation mode
    total_metrics = {metric: 0.0 for metric in ae_eval_config.get('metrics', ["psnr", "ssim"])}
    num_samples = 0 # Counter for total samples processed
    samples_saved = 0 # Counter for images saved
    start_time = time.time()

    logger.info("Starting AE reconstruction evaluation...")

    with torch.no_grad(): # Disable gradient calculation
        # The dataloader now yields (ndct_image, filename)
        for batch_idx, (ndct_images, filenames) in enumerate(eval_dataloader):
            ndct_images = ndct_images.to(device)

            # Forward pass through the AE
            reconstructed_images = model(ndct_images)

            # Calculate metrics and save images for each sample in the batch
            for i in range(ndct_images.size(0)):
                img_ndct = ndct_images[i] # (C, H, W) tensor
                img_recon = reconstructed_images[i] # (C, H, W) tensor
                filename = filenames[i] # Corresponding filename

                # Calculate specified metrics
                if "psnr" in total_metrics:
                    # PSNR expects images in [0, 255] range (after conversion from [0, 1] float)
                    psnr_val = calculate_psnr(img_recon, img_ndct, data_range=1.0)
                    total_metrics["psnr"] += psnr_val
                if "ssim" in total_metrics:
                    # calculate_ssim expects (N, C, H, W), so add batch dim
                    ssim_val = calculate_ssim(img_recon.unsqueeze(0), img_ndct.unsqueeze(0), data_range=1.0)
                    total_metrics["ssim"] += ssim_val

                num_samples += 1

                # Save reconstructed images if output_dir is specified AND
                # if we haven't reached the save limit (if set)
                if output_dir and (save_num_images < 0 or samples_saved < save_num_images):
                    try:
                        # Convert tensor to numpy uint8 [0, 255] for saving
                        img_recon_np = tensor2numpy(img_recon, rgb_range=1.0)
                        # Modify filename to indicate it's a reconstructed image
                        save_filename = filename.replace('.png', '_recon.png')
                        save_path = os.path.join(output_dir, save_filename)
                        Image.fromarray(img_recon_np).save(save_path)
                        samples_saved += 1
                        if samples_saved % 10 == 0: # Log saving progress periodically
                            logger.info(f"Saved {samples_saved} reconstructed images so far.")
                    except Exception as e:
                        logger.error(f"Error saving reconstructed image {filename}: {e}")


            # Stop processing batches if we've saved enough images and a limit was set
            if save_num_images >= 0 and samples_saved >= save_num_images:
                 logger.info(f"Saved {save_num_images} images. Stopping batch processing.")
                 break # Exit the batch loop


    # Calculate average metrics
    # Note: If we broke early due to save_num_images, the metrics will only be over the processed samples
    avg_metrics = {metric: total / num_samples if num_samples > 0 else 0.0 for metric, total in total_metrics.items()}

    elapsed_time = time.time() - start_time
    logger.info(f"AE Evaluation finished in {elapsed_time:.2f}s. Average metrics over {num_samples} samples:")
    for metric, avg_val in avg_metrics.items():
        logger.info(f"  Avg {metric.upper()}: {avg_val:.4f}")

    # You can return the average metrics if needed by another script
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the trained Autoencoder model.')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the main configuration file (YAML). '
                             'Uses the "autoencoder" and "ae_evaluation" sections.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the trained Autoencoder model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default='results/ae_eval',
                        help='Directory to save reconstructed images and evaluation log.')
    parser.add_argument('--save_num_images', type=int, default=-1,
                        help='Number of reconstructed images to save. Set to -1 to save all.')


    args = parser.parse_args()

    # Ensure the checkpoint path from args is used, overriding the config file's model_path
    # The config's model_path is primarily for the CGNet training script to find the AE.
    # For evaluation, the user explicitly provides the checkpoint path.
    # We could potentially update the config's model_path here for consistency if desired,
    # but it's safer to just use the --checkpoint argument directly for evaluation.
    # config['autoencoder']['model_path'] = args.checkpoint # Optional: update config

    main(args.config, args.checkpoint, args.output_dir, args.save_num_images)

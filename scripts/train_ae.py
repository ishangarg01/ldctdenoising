# scripts/train_ae.py

import argparse
import yaml
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from datetime import datetime
import shutil # Import shutil for copying config
import logging # Import logging

# Import custom modules
# Assuming datasets.ct_denoise_dataset is available relative to the project root
from datasets.ct_denoise_dataset import CTDenoiseDataset
# NOTE: Replace the placeholder define_autoencoder below with the actual import
# from models.autoencoder import define_autoencoder
# Assuming losses.combined_loss is available relative to the project root
# The AE training script uses a simple reconstruction loss (MSE or MAE), not CombinedLoss
# from losses.combined_loss import CombinedLoss # Not needed for AE training
# Assuming utils.logging_utils is available relative to the project root
from utils.logging_utils import setup_logger
# Assuming utils.metrics is available relative to the project root
from utils.metrics import calculate_psnr, calculate_ssim # Metrics used in evaluate_ae
# Assuming utils.eval_utils is available relative to the project root
# The evaluate_ae function is defined within this script, not imported from eval_utils
# from utils.eval_utils import evaluate_model # Not needed here

# Setup logger for this script
# Configure the root logger or a specific logger early
# This ensures a basic logger is available even before main() is called
# The logger inside main() will then get and potentially reconfigure this logger or a new one
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Use the utility to setup if not already configured by main execution
    setup_logger(__name__) # Setup a basic console logger initially


# Placeholder for define_autoencoder - REMOVE THIS ONCE models.autoencoder.__init__.py IS READY
# This should be replaced with the actual import from models.autoencoder.__init__.py.
# It currently returns a dummy AE model for testing purposes.
def define_autoencoder(opt):
    """
    Placeholder function to define AE.
    This should be replaced with the actual import from models.autoencoder.__init__.py.
    It currently returns a dummy AE model for testing purposes.
    """
    logger.info(f"Using placeholder define_autoencoder. Defining dummy AE of type: {opt['arch']['type']} with args: {opt['arch'].get('args', {})}")
    # Placeholder: Return a dummy AE model mimicking the expected interface
    class DummyAE(nn.Module):
        def __init__(self, in_channels, base_channels, num_encoder_layers):
            super().__init__()
            # Simple placeholder encoder/decoder structure
            # This structure should match the expected input/output sizes for feature extraction
            # based on the number of encoder layers (downsampling steps)
            layers = []
            current_channels = in_channels
            for i in range(num_encoder_layers):
                 out_channels = base_channels * (2 ** i)
                 layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
                 layers.append(nn.ReLU(inplace=True))
                 current_channels = out_channels
            self.encoder = nn.Sequential(*layers)

            # Decoder needs to roughly mirror the encoder for reconstruction
            decoder_layers = []
            current_channels = base_channels * (2 ** (num_encoder_layers - 1)) if num_encoder_layers > 0 else in_channels
            for i in range(num_encoder_layers - 1, -1, -1):
                 out_channels = base_channels * (2 ** (i - 1)) if i > 0 else in_channels
                 decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
                 decoder_layers.append(nn.ReLU(inplace=True) if i > 0 else nn.Identity())
                 current_channels = out_channels

            # Final conv layer and sigmoid for output
            decoder_layers.append(nn.Conv2d(current_channels, in_channels, kernel_size=3, stride=1, padding=1))
            self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid())

            # Ensure get_features method exists for consistency with the planned interface
        def forward(self, x):
            # Full AE forward pass for reconstruction training
            return self.decoder(self.encoder(x))

        def get_features(self, x):
             # Method to get features from the encoder for perceptual loss
             return self.encoder(x) # Return encoder output as features

    # Instantiate dummy AE based on placeholder args from the config's 'arch' section
    ae_arch_args = opt.get('arch', {}).get('args', {})
    # Provide default values if args are missing in the config
    return DummyAE(
        in_channels=ae_arch_args.get('in_channels', 3),
        base_channels=ae_arch_args.get('base_channels', 16),
        num_encoder_layers=ae_arch_args.get('num_encoder_layers', 3) # Default to 3 layers as planned
    )
# END OF PLACEHOLDER


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            # Set seeds for CUDA if available.
            # Note: Setting deterministic and benchmark to False can impact performance
            # but ensures bit-for-bit reproducibility across runs with the same setup.
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Loads a model checkpoint and returns the epoch to resume from.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load the state dict into.
        optimizer (Optimizer, optional): The optimizer to load the state dict into.
        scheduler (Scheduler, optional): The scheduler to load the state dict into.

    Returns:
        int: The epoch number to resume training from (next epoch after saved).
             Returns 0 if the checkpoint is not found.
    """
    if not os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint not found: {checkpoint_path}. Starting from epoch 0.")
        return 0 # Start from epoch 0

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    # Load checkpoint to CPU first to avoid potential CUDA issues when loading
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state dict loaded.")
    except (RuntimeError, ValueError) as e:
        logger.error(f"Error loading model state dict from {checkpoint_path}: {e}")
        logger.warning("Model state dict mismatch. Starting training without loading model weights.")
        # Decide whether to exit or continue without loading model weights
        # For now, we'll let it continue, but this might not be desired behavior.
        # You might want to raise the error or exit here in a production system.


    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state dict loaded.")
        except ValueError:
            logger.warning("Optimizer state dict mismatch. Skipping optimizer state loading.")
    else:
        logger.info("Optimizer state dict not found in checkpoint or optimizer not provided.")


    if scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state dict loaded.")
        except ValueError:
            logger.warning("Scheduler state dict mismatch. Skipping scheduler state loading.")
    else:
        logger.info("Scheduler state dict not found in checkpoint or scheduler not provided.")


    start_epoch = checkpoint.get('epoch', 0) + 1 # Start from the next epoch
    logger.info(f"Resuming training from epoch {start_epoch}")
    return start_epoch

def save_checkpoint(epoch, model, optimizer, scheduler, loss, checkpoint_dir, experiment_name, is_best=False):
    """
    Saves a model checkpoint.

    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer to save.
        scheduler (Scheduler, optional): The scheduler to save.
        loss (float): The current loss value (e.g., average epoch loss).
        checkpoint_dir (str): The directory to save checkpoints.
        experiment_name (str): The name of the experiment.
        is_best (bool): Whether this is the best model so far (saves an additional 'best' checkpoint).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the epoch-specific checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_epoch_{epoch:03d}.pth')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    if scheduler:
        state['scheduler_state_dict'] = scheduler.state_dict()

    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        # Also update the 'latest' checkpoint symlink or copy
        latest_path = os.path.join(checkpoint_dir, f'{experiment_name}_latest.pth')
        # Use copyfile for simplicity, symlink might be preferred on some systems
        shutil.copyfile(checkpoint_path, latest_path)
        logger.info(f"Copied latest checkpoint to {latest_path}")

        # Optionally save a 'best' checkpoint if applicable (requires tracking best metric)
        if is_best:
            best_path = os.path.join(checkpoint_dir, f'{experiment_name}_best.pth')
            shutil.copyfile(checkpoint_path, best_path)
            logger.info(f"Copied best checkpoint to {best_path}")

    except Exception as e:
        logger.error(f"Error saving checkpoint {checkpoint_path}: {e}")


def evaluate_ae(model, dataloader, device, logger_eval, epoch=None, writer=None, metrics_list=["psnr", "ssim"]):
    """
    Evaluates the Autoencoder's reconstruction performance on a dataset.

    Args:
        model (nn.Module): The trained AE model.
        dataloader (DataLoader): DataLoader for the evaluation dataset (NDCT images).
        device (torch.device): Device to perform evaluation on.
        logger_eval (logging.Logger): Logger for evaluation output.
        epoch (int, optional): Current epoch number for logging/TensorBoard.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter.
        metrics_list (list): List of metric names to calculate (e.g., ["psnr", "ssim"]).

    Returns:
        dict: Dictionary of average metric values.
    """
    model.eval() # Set model to evaluation mode
    total_metrics = {metric: 0.0 for metric in metrics_list}
    num_samples = 0
    start_time = time.time()

    logger_eval.info(f"Starting AE evaluation{f' for epoch {epoch}' if epoch is not None else ''}...")

    with torch.no_grad(): # Disable gradient calculation
        # CTDenoiseDataset returns (LDCT, NDCT), but NDCTOnlyDataset (used here) returns only NDCT
        for batch_idx, ndct_images in enumerate(dataloader):
            ndct_images = ndct_images.to(device)

            # Forward pass through the AE
            reconstructed_images = model(ndct_images)

            # Calculate metrics for each image in the batch
            for i in range(ndct_images.size(0)):
                img_ndct = ndct_images[i] # (C, H, W) tensor
                img_recon = reconstructed_images[i] # (C, H, W) tensor

                # Calculate specified metrics
                if "psnr" in metrics_list:
                    # PSNR expects images in [0, 255] range (after conversion from [0, 1] float)
                    # calculate_psnr handles the conversion internally if needed based on data_range
                    psnr_val = calculate_psnr(img_recon, img_ndct, data_range=1.0) # Pass data_range=1.0 to the metric function
                    total_metrics["psnr"] += psnr_val
                if "ssim" in metrics_list:
                    # calculate_ssim expects (N, C, H, W), so add batch dim
                    ssim_val = calculate_ssim(img_recon.unsqueeze(0), img_ndct.unsqueeze(0), data_range=1.0) # Pass data_range=1.0
                    total_metrics["ssim"] += ssim_val

                num_samples += 1

            # Log progress if needed (optional for eval, can be verbose)
            # if (batch_idx + 1) % log_interval == 0:
            #      logger_eval.info(f"Eval Batch [{batch_idx+1}/{len(dataloader)}]")


    # Calculate average metrics
    avg_metrics = {metric: total / num_samples if num_samples > 0 else 0.0 for metric, total in total_metrics.items()}

    elapsed_time = time.time() - start_time
    logger_eval.info(f"AE Evaluation finished in {elapsed_time:.2f}s. Average metrics over {num_samples} samples:")
    for metric, avg_val in avg_metrics.items():
        logger_eval.info(f"  Avg {metric.upper()}: {avg_val:.4f}")
        # Log to TensorBoard if writer is provided and epoch is not None
        if writer is not None and epoch is not None:
             writer.add_scalar(f'AE_Eval/{metric.upper()}', avg_val, epoch)

    model.train() # Set model back to training mode
    return avg_metrics


def main(ae_config_path, main_config_path='configs/default_config.yaml'):
    """Main training function for the Autoencoder."""
    # Load AE specific config
    with open(ae_config_path, 'r') as f:
        ae_config = yaml.safe_load(f)

    # Load main config to get the target path for the AE checkpoint
    # Do this early, before setting up the run directory and logger
    try:
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)
        main_ae_model_path = main_config['autoencoder']['model_path']
        # We can't log this using the run-specific logger yet, but we can print or use the basic logger
        logging.info(f"Target AE checkpoint path from main config: {main_ae_model_path}")
    except FileNotFoundError:
        logging.error(f"Main config file not found at {main_config_path}. Cannot determine target AE checkpoint path.")
        raise # Re-raise the error
    except KeyError:
        logging.error(f"'autoencoder.model_path' not found in main config file {main_config_path}.")
        raise # Re-raise the error


    # Use the ae_training section of the config
    ae_training_config = ae_config['ae_training']
    ae_model_config = ae_config['autoencoder'] # Get AE architecture config from AE config file
    dataset_config = ae_config['dataset'] # Get dataset config from AE config file
    ae_evaluation_config = ae_config['ae_evaluation'] # Get AE evaluation config from AE config file


    # --- Setup ---
    # Set random seed for reproducibility
    set_seed(ae_training_config.get('seed', 42)) # Use seed from AE config, default to 42

    # Create experiment directory
    experiment_name = ae_training_config.get('experiment_name', 'ae_train_run')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Default output directory for AE runs is inside experiments/ or data/processed/
    # Let's use data/processed/ae_runs for AE checkpoints as they are prerequisites
    # for CGNet training and might be stored alongside processed data.
    default_ae_output_dir = os.path.join('data', 'processed', 'ae_runs')
    run_dir = os.path.join(ae_training_config.get('output_dir', default_ae_output_dir), f'{experiment_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # --- Setup Logging (NOW it's safe to set up the run-specific logger) ---
    log_file = os.path.join(run_dir, 'train_ae.log')
    # Pass run_dir to logger setup if it needs to create the directory
    # Get the logger instance again within main to ensure it's the local variable used
    logger = setup_logger('train_ae_logger', log_file=log_file, level=logging.INFO)
    logger.info(f"Starting AE training experiment: {experiment_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"AE Config:\n{yaml.dump(ae_config, indent=4)}") # Log the AE config
    logger.info(f"Main Config (for AE path): {main_config_path}") # Log the main config path used
    # Now it's safe to log the target path using the run-specific logger
    logger.info(f"Target AE checkpoint path from main config: {main_ae_model_path}")


    # Save the AE config file used for this run into the experiment directory
    try:
        shutil.copyfile(ae_config_path, os.path.join(run_dir, 'config.yaml'))
    except shutil.SameFileError:
        pass # Handle case where config is already in run_dir if output_dir is config dir


    # Setup TensorBoard SummaryWriter
    tensorboard_log_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard logs saving to: {tensorboard_log_dir}")


    # Device setup
    device = torch.device(ae_training_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")


    # --- Data Loading ---
    # For AE training, we only need the full-dose (NDCT) images from the training split
    # CTDenoiseDataset loads pairs (LDCT, NDCT), so we'll use a wrapper dataset
    # to only return the target (full-dose) image.
    train_dataset = CTDenoiseDataset(
        root=dataset_config['args']['root'],
        mode='train',
        transform=None # Use default ToTensor transform from CTDenoiseDataset
    )
    # Wrapper dataset to extract only the NDCT image (target)
    class NDCTOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, idx):
            # CTDenoiseDataset returns (quarter_img, full_img)
            _, full_img = self.base_dataset[idx]
            return full_img # Return only the full_img (NDCT)

    train_ndct_dataset = NDCTOnlyDataset(train_dataset)

    train_dataloader = DataLoader(
        train_ndct_dataset,
        batch_size=dataset_config.get('train_batch_size', 8), # Use train_batch_size from dataset config
        shuffle=True,
        num_workers=dataset_config.get('num_workers', 4), # Use num_workers from dataset config
        pin_memory=True and device.type == 'cuda' # Pin memory only if using CUDA
    )
    logger.info(f"Train DataLoader created with {len(train_dataloader)} batches.")


    # Optional: Validation dataloader for evaluation during training
    # Use the test split's full-dose images for validation
    # This helps monitor AE reconstruction performance on unseen data during training
    eval_dataset = CTDenoiseDataset(
         root=dataset_config['args']['root'],
         mode='test', # Use test data for evaluation
         transform=None # Use default ToTensor transform
    )
    eval_ndct_dataset = NDCTOnlyDataset(eval_dataset)

    eval_dataloader = DataLoader(
         eval_ndct_dataset,
         batch_size=dataset_config.get('test_batch_size', 1), # Use test_batch_size
         shuffle=False, # No need to shuffle evaluation data
         num_workers=dataset_config.get('num_workers', 4),
         pin_memory=True and device.type == 'cuda' # Pin memory only if using CUDA
    )
    logger.info(f"Eval DataLoader created with {len(eval_dataloader)} batches.")


    # --- Model Setup ---
    # Define the Autoencoder model using the AE architecture config from the AE config file
    # NOTE: Ensure models.autoencoder.__init__.py and the AE architecture file (e.g., simple_conv_ae.py) are ready
    try:
         # Attempt to import the actual define_autoencoder
         from models.autoencoder import define_autoencoder as actual_define_autoencoder
         model = actual_define_autoencoder(ae_model_config).to(device)
         logger.info("Successfully imported and defined Autoencoder model.")
    except ImportError:
         logger.error("Could not import actual define_autoencoder from models.autoencoder. Using placeholder.")
         # Fallback to placeholder if actual import fails (for initial testing)
         model = define_autoencoder(ae_model_config).to(device)
    except Exception as e:
         logger.error(f"Error defining Autoencoder model: {e}. Using placeholder.")
         model = define_autoencoder(ae_model_config).to(device)


    logger.info(f"Autoencoder Model:\n{model}")


    # --- Loss, Optimizer, Scheduler ---
    # Reconstruction loss for AE training (MSE or MAE)
    loss_type = ae_training_config.get('loss_type', 'mse')
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        logger.info("Using MSELoss for AE reconstruction.")
    elif loss_type == 'mae':
        criterion = nn.L1Loss()
        logger.info("Using L1Loss for AE reconstruction.")
    else:
        raise ValueError(f"Unsupported AE loss_type: {loss_type}. Choose 'mse' or 'mae'.")

    optimizer_config = ae_training_config.get('optimizer', {'type': 'Adam', 'args': {'lr': 0.001}})
    # Ensure optimizer type exists in torch.optim
    if not hasattr(torch.optim, optimizer_config['type']):
         raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    optimizer = torch.optim.__dict__[optimizer_config['type']](
        model.parameters(), **optimizer_config['args'])
    logger.info(f"Optimizer: {optimizer}")


    scheduler = None
    scheduler_config = ae_training_config.get('scheduler')
    if scheduler_config:
        # Ensure scheduler type exists in torch.optim.lr_scheduler
        if not hasattr(torch.optim.lr_scheduler, scheduler_config['type']):
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler_config['type']](
            optimizer, **scheduler_config['args'])
        logger.info(f"Scheduler: {scheduler}")


    # --- Resume Training (Optional) ---
    start_epoch = 0
    resume_checkpoint_path = ae_training_config.get('resume_checkpoint')
    if resume_checkpoint_path:
        # If resume_checkpoint_path is relative, make it relative to the run_dir
        if not os.path.isabs(resume_checkpoint_path):
             # Assuming relative paths are relative to the run directory for resumption
             resume_checkpoint_path = os.path.join(run_dir, resume_checkpoint_path)
        start_epoch = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler)


    # --- Training Loop ---
    logger.info("Starting AE training...")
    # Track the best evaluation metric for saving the 'best' checkpoint
    # Initialize based on the first metric in the eval list, assuming higher is better
    eval_metrics_list = ae_evaluation_config.get('metrics', ["psnr"]) # Default to PSNR if not specified
    if not eval_metrics_list:
         logger.warning("No evaluation metrics specified for AE. Cannot track 'best' model.")
         primary_eval_metric_name = None
         best_eval_metric = None # No metric to track
    else:
        primary_eval_metric_name = eval_metrics_list[0]
        # Assuming PSNR/SSIM where higher is better. Adjust if using a loss or different metric.
        best_eval_metric = float('-inf')
        if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
             best_eval_metric = float('inf')


    for epoch in range(start_epoch, ae_training_config['epochs']):
        model.train() # Set model to training mode
        total_loss = 0
        epoch_start_time = time.time()

        logger.info(f"Epoch {epoch}/{ae_training_config['epochs']} starting...")

        # Use tqdm for progress bar in console if available (optional)
        # try:
        #     from tqdm import tqdm
        #     dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        # except ImportError:
        #     dataloader_iter = train_dataloader
        dataloader_iter = train_dataloader # Use standard iterator


        for batch_idx, inputs in enumerate(dataloader_iter):
            # inputs are the NDCT images from NDCTOnlyDataset
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, inputs) # AE reconstruction loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log training progress
            if (batch_idx + 1) % ae_training_config.get('log_interval', 50) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                # elapsed_time = time.time() - epoch_start_time # Time since epoch start
                logger.info(f"Epoch [{epoch}/{ae_training_config['epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}") # Removed time for brevity


                # Log to TensorBoard
                global_step = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('AE_Train/batch_loss', loss.item(), global_step)
                writer.add_scalar('AE_Train/avg_loss', avg_loss, global_step)
                writer.add_scalar('AE_Train/lr', optimizer.param_groups[0]['lr'], global_step)


        # End of Epoch
        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f}. Epoch time: {epoch_elapsed_time:.2f}s")

        # Log epoch loss to TensorBoard
        writer.add_scalar('AE_Train/epoch_loss', avg_epoch_loss, epoch)

        # Step the scheduler
        if scheduler:
            scheduler.step()
            logger.info(f"Scheduler stepped. New LR: {optimizer.param_groups[0]['lr']:.6f}")


        # Save checkpoint periodically
        save_interval = ae_training_config.get('save_interval', 10)
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
             save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name)

        # Evaluate on validation set periodically
        eval_interval = ae_training_config.get('eval_interval', -1)
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0 and len(eval_dataloader) > 0:
            avg_eval_metrics = evaluate_ae(model, eval_dataloader, device, logger, epoch, writer, ae_evaluation_config.get('metrics', ["psnr", "ssim"]))

            # Check if this is the best model based on the primary evaluation metric
            if primary_eval_metric_name and primary_eval_metric_name in avg_eval_metrics:
                 current_metric_value = avg_eval_metrics[primary_eval_metric_name]
                 # Assuming PSNR/SSIM where higher is better
                 is_best = current_metric_value > best_eval_metric
                 if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
                      is_best = current_metric_value < best_eval_metric

                 if is_best:
                      best_eval_metric = current_metric_value
                      logger.info(f"New best AE model found based on {primary_eval_metric_name}: {best_eval_metric:.4f}. Saving checkpoint.")
                      # Save checkpoint and mark as best
                      save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name, is_best=True)
                 else:
                      logger.info(f"Current {primary_eval_metric_name}: {current_metric_value:.4f}, Best {primary_eval_metric_name}: {best_eval_metric:.4f}")
            elif primary_eval_metric_name:
                 logger.warning(f"Primary evaluation metric '{primary_eval_metric_name}' not found in evaluation results.")


    # --- End Training ---
    logger.info("AE training finished.")
    writer.close()

    # Save final model checkpoint (this saves epoch_XXX.pth and copies to _latest.pth)
    final_epoch = ae_training_config['epochs'] - 1
    # Call save_checkpoint one last time to ensure the final epoch's state is saved and _latest.pth is updated
    save_checkpoint(final_epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name)
    # The actual path of the 'latest' checkpoint saved by save_checkpoint
    latest_checkpoint_path_in_run_dir = os.path.join(run_dir, f'{experiment_name}_latest.pth')
    logger.info(f"Latest AE checkpoint in run directory: {latest_checkpoint_path_in_run_dir}")


    # Copy the final/latest checkpoint to the path specified in the main config for CGNet training
    # This path is where the CGNet training script will look for the pre-trained AE
    # This is the fix: using the main_ae_model_path obtained from default_config.yaml
    try:
        os.makedirs(os.path.dirname(main_ae_model_path), exist_ok=True)
        # Copy the _latest.pth file instead of the non-existent _final.pth
        shutil.copyfile(latest_checkpoint_path_in_run_dir, main_ae_model_path)
        logger.info(f"Copied final AE checkpoint ({latest_checkpoint_path_in_run_dir}) to main config path: {main_ae_model_path}")
    except Exception as e:
        logger.error(f"Error copying final AE checkpoint to {main_ae_model_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Autoencoder model.')
    parser.add_argument('--config', type=str, default='configs/ae_config.yaml',
                        help='Path to the AE configuration file (YAML).')
    # Add argument for the main config path, default to the standard location
    parser.add_argument('--main_config', type=str, default='configs/default_config.yaml',
                        help='Path to the main configuration file (YAML) to get AE model_path.')
    args = parser.parse_args()
    # Pass both config paths to the main function
    main(args.config, args.main_config)

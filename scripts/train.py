# # scripts/train.py

# import argparse
# import yaml
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import random
# from datetime import datetime
# import shutil # Import shutil for copying config
# import logging # Import logging

# # Import custom modules
# from datasets.ct_denoise_dataset import CTDenoiseDataset # Assuming this is correct

# # --- Import actual modules ---
# # Uncommenting these lines to attempt importing the real functions/classes.
# # If any of these imports fail, an ImportError will be raised.
# # Added specific try...except blocks for better error reporting.

# # Import logging utility first
# try:
#     from utils.logging_utils import setup_logger
#     # Setup logger for this script using the imported utility
#     logger = logging.getLogger(__name__)
#     # Set logger level to INFO for less verbose logging during normal runs
#     # The setup_logger in main will configure the file handler specifically.
#     if not logger.handlers:
#         # Basic config for console output before main is called
#         setup_logger(__name__, level=logging.INFO) # Set initial console logger to INFO
#     else:
#         # If handlers exist, ensure level is set to INFO
#         for handler in logger.handlers:
#             handler.setLevel(logging.INFO)


#     logger.info("Successfully imported and configured logging.")
# except ImportError as e:
#     # Fallback to basic logging if setup_logger cannot be imported
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Fallback to INFO
#     logger = logging.getLogger(__name__) # Get logger again after basic config
#     logger.error(f"ImportError: Could not import setup_logger from utils.logging_utils: {e}")
#     logger.error("Using basic logging configuration at INFO level.")
# except Exception as e:
#      logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Fallback to INFO
#      logger = logging.getLogger(__name__)
#      logger.error(f"Error configuring logging: {e}")
#      logger.error("Using basic logging configuration at INFO level.")


# # Now import other modules with specific error logging.
# try:
#     from models.archs import define_network
#     logger.debug("Successfully imported define_network from models.archs.")
# except ImportError as e:
#     logger.error(f"ImportError: Could not import define_network from models.archs: {e}")
#     logger.error("Please check models/archs/__init__.py and your model architecture files.")
#     raise # Re-raise the error as the model is essential
# except Exception as e:
#     logger.error(f"Error importing define_network: {e}")
#     raise # Re-raise the error

# try:
#     from models.autoencoder import define_autoencoder
#     logger.debug("Successfully imported define_autoencoder from models.autoencoder.")
# except ImportError as e:
#     logger.error(f"ImportError: Could not import define_autoencoder from models.autoencoder: {e}")
#     logger.error("Please check models/autoencoder/__init__.py and your AE architecture files.")
#     raise # Re-raise the error as the autoencoder is essential
# except Exception as e:
#     logger.error(f"Error importing define_autoencoder: {e}")
#     raise # Re-raise the error

# try:
#     from losses.combined_loss import CombinedLoss
#     logger.debug("Successfully imported CombinedLoss from losses.combined_loss.")
# except ImportError as e:
#     logger.error(f"ImportError: Could not import CombinedLoss from losses.combined_loss: {e}")
#     logger.error("Please check losses/combined_loss.py and its dependencies (like learnable_weights.py or pytorch-msssim).")
#     raise # Re-raise the error as the loss is essential
# except Exception as e:
#     logger.error(f"Error importing CombinedLoss: {e}")
#     raise # Re-raise the error

# # Add specific error logging for the eval_utils import
# try:
#     from utils.eval_utils import evaluate_model
#     logger.debug("Successfully imported evaluate_model from utils.eval_utils.")
# except ImportError as e:
#     logger.error(f"ImportError: Could not import evaluate_model from utils.eval_utils: {e}")
#     logger.error("This is likely causing the 'Using placeholder evaluate_model' message if it appears.")
#     logger.error("Please check utils/eval_utils.py and its dependencies (like utils/metrics.py or utils/logging_utils.py).")
#     logger.error("Verify file paths and project structure relative to where you are running train.py.")
#     raise # Re-raise the error to stop execution with a clear message
# except Exception as e:
#      logger.error(f"Error importing evaluate_model: {e}")
#      logger.error("This is likely causing the 'Using placeholder evaluate_model' message if it appears.")
#      raise # Re-raise the error

# # Import metrics functions (used by evaluate_model)
# try:
#     from utils.metrics import calculate_psnr, calculate_ssim
#     logger.debug("Successfully imported metrics functions from utils.metrics.")
# except ImportError as e:
#     logger.error(f"ImportError: Could not import metrics functions from utils.metrics: {e}")
#     logger.error("Please check utils/metrics.py and its dependencies (like scikit-image or pytorch-msssim).")
#     raise # Re-raise the error if metrics cannot be imported
# except Exception as e:
#     logger.error(f"Error importing metrics functions: {e}")
#     raise # Re-raise the error


# # --- REMOVED PLACEHOLDER DEFINITIONS ---
# # The placeholder functions for define_network, define_autoencoder, CombinedLoss,
# # and evaluate_model have been removed.
# # If the imports above fail, the script will now raise an ImportError immediately.


# def set_seed(seed):
#     """Set random seeds for reproducibility."""
#     if seed is not None:
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
#             # torch.backends.cudnn.deterministic = True # Can slow down training
#             # torch.backends.cudnn.benchmark = False    # Can slow down training


# def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, criterion=None):
#     """
#     Loads a model checkpoint and returns the epoch to resume from.

#     Args:
#         checkpoint_path (str): Path to the checkpoint file.
#         model (nn.Module): The model to load the state dict into.
#         optimizer (Optimizer, optional): The optimizer to load the state dict into.
#         scheduler (Scheduler, optional): The scheduler to load the state dict into.
#         criterion (nn.Module, optional): The criterion (CombinedLoss) to load the state dict into
#                                          if it uses learnable weights.

#     Returns:
#         int: The epoch number to resume training from (next epoch after saved).
#              Returns 0 if the checkpoint is not found.
#     """
#     if not os.path.exists(checkpoint_path):
#         logger.info(f"Checkpoint not found: {checkpoint_path}. Starting from epoch 0.")
#         return 0 # Start from epoch 0

#     logger.info(f"Loading checkpoint: {checkpoint_path}")
#     # Load checkpoint to CPU first to avoid potential CUDA issues when loading
#     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#     try:
#         model.load_state_dict(checkpoint['model_state_dict'])
#         logger.info("Model state dict loaded.")
#     except (RuntimeError, ValueError) as e:
#         logger.error(f"Error loading model state dict from {checkpoint_path}: {e}")
#         logger.warning("Model state dict mismatch. Starting training without loading model weights.")
#         # Decide whether to exit or continue without loading model weights
#         # For now, we'll let it continue, but this might not be desired behavior.
#         # You might want to raise the error or exit here in a production system.


#     if optimizer and 'optimizer_state_dict' in checkpoint:
#         try:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             logger.info("Optimizer state dict loaded.")
#         except ValueError:
#             logger.warning("Optimizer state dict mismatch. Skipping optimizer state loading.")
#     else:
#         logger.info("Optimizer state dict not found in checkpoint or optimizer not provided.")


#     if scheduler and 'scheduler_state_dict' in checkpoint:
#         try:
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#             logger.info("Scheduler state dict loaded.")
#         except ValueError:
#             logger.warning("Scheduler state dict mismatch. Skipping scheduler state loading.")
#     else:
#         logger.info("Scheduler state dict not found in checkpoint or scheduler not provided.")


#     # If using learnable weights, load the criterion's state dict
#     # Check if criterion is provided and if it's using learnable weights
#     if criterion is not None and hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
#        hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
#          try:
#              if 'criterion_state_dict' in checkpoint:
#                   criterion.load_state_dict(checkpoint['criterion_state_dict'])
#                   logger.info("Criterion state dict loaded.")
#              else:
#                   logger.warning("Criterion state dict not found in checkpoint. Learnable weights will start from initial values.")
#          except Exception as e:
#               logger.warning(f"Could not load criterion state dict from {checkpoint_path}: {e}. Learnable weights will start from initial values.")


#     start_epoch = checkpoint.get('epoch', 0) + 1 # Start from the next epoch
#     logger.info(f"Resuming training from epoch {start_epoch}")
#     return start_epoch

# def save_checkpoint(epoch, model, optimizer, scheduler, loss, checkpoint_dir, experiment_name, criterion=None, is_best=False):
#     """
#     Saves a model checkpoint.

#     Args:
#         epoch (int): The current epoch number.
#         model (nn.Module): The model to save.
#         optimizer (Optimizer): The optimizer to save.
#         scheduler (Scheduler, optional): The scheduler to save.
#         loss (float): The current loss value (e.e., average epoch loss).
#         checkpoint_dir (str): The directory to save checkpoints.
#         experiment_name (str): The name of the experiment.
#         criterion (nn.Module, optional): The criterion (CombinedLoss) to save the state dict for
#                                          if it uses learnable weights.
#         is_best (bool): Whether this is the best model so far (saves an additional 'best' checkpoint).
#     """
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_epoch_{epoch:03d}.pth')
#     state = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
#     }
#     if scheduler:
#         state['scheduler_state_dict'] = scheduler.state_dict()

#     # If using learnable weights, save the criterion's state dict
#     # Check if criterion is provided and if it's using learnable weights
#     if criterion is not None and hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
#        hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
#          state['criterion_state_dict'] = criterion.state_dict()
#          logger.debug("Saving criterion state dict (including learnable weights).")


#     try:
#         torch.save(state, checkpoint_path)
#         logger.info(f"Checkpoint saved to {checkpoint_path}")
#         # Optionally save a 'latest' checkpoint symlink or copy
#         latest_path = os.path.join(checkpoint_dir, f'{experiment_name}_latest.pth')
#         # Use copyfile for simplicity, symlink might be preferred on some systems
#         shutil.copyfile(checkpoint_path, latest_path)
#         logger.info(f"Copied latest checkpoint to {latest_path}")

#         # Optionally save a 'best' checkpoint if applicable (requires tracking best metric)
#         if is_best:
#             best_path = os.path.join(checkpoint_dir, f'{experiment_name}_best.pth')
#             shutil.copyfile(checkpoint_path, best_path)
#             logger.info(f"Copied best checkpoint to {best_path}")

#     except Exception as e:
#         logger.error(f"Error saving checkpoint {checkpoint_path}: {e}")


# def main(config_path):
#     """Main training function for the CGNet model."""
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # Use the relevant sections of the config
#     model_config = config['model']
#     autoencoder_config = config['autoencoder']
#     dataset_config = config['dataset']
#     loss_config = config['loss']
#     optimizer_config = config['optimizer']
#     scheduler_config = config.get('scheduler') # Optional
#     training_config = config['training']
#     evaluation_config = config['evaluation']


#     # --- Setup ---
#     # Set random seed for reproducibility
#     set_seed(training_config.get('seed', 42))

#     # Create experiment directory
#     experiment_name = training_config.get('experiment_name', 'cgnet_train_run')
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     # Output directory for this specific run
#     run_dir = os.path.join(training_config.get('output_dir', 'experiments'), f'{experiment_name}_{timestamp}')
#     os.makedirs(run_dir, exist_ok=True)

#     # Save the config file used for this run into the experiment directory
#     try:
#         shutil.copyfile(config_path, os.path.join(run_dir, 'config.yaml'))
#     except shutil.SameFileError:
#         pass # Handle case where config is already in run_dir if output_dir is config dir


#     # Setup logging - use the utility function
#     log_file = os.path.join(run_dir, 'train.log')
#     # Use the imported setup_logger
#     # Configure train_logger to write to the file with INFO level
#     logger = setup_logger('train_logger', log_file=log_file, level=logging.INFO) # Set train_logger to INFO
#     logger.info(f"Starting CGNet training experiment: {experiment_name}")
#     logger.info(f"Run directory: {run_dir}")
#     logger.info(f"Config:\n{yaml.dump(config, indent=4)}") # Log the full config


#     # Setup TensorBoard SummaryWriter
#     tensorboard_log_dir = os.path.join(run_dir, 'tensorboard')
#     writer = SummaryWriter(log_dir=tensorboard_log_dir)
#     logger.info(f"TensorBoard logs saving to: {tensorboard_log_dir}")


#     # Device setup
#     device = torch.device(training_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")


#     # --- Data Loading ---
#     # Load training dataset and dataloader
#     train_dataset = CTDenoiseDataset(
#         root=dataset_config['args']['root'],
#         mode='train',
#         transform=None # Use default ToTensor transform
#     )
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=dataset_config.get('train_batch_size', 8),
#         shuffle=True,
#         num_workers=dataset_config.get('num_workers', 4),
#         pin_memory=True and device.type == 'cuda'
#     )
#     logger.info(f"Train DataLoader created with {len(train_dataloader)} batches.")


#     # Load evaluation dataset and dataloader (using test split)
#     eval_dataset = CTDenoiseDataset(
#         root=dataset_config['args']['root'],
#         mode='test', # Evaluate on the test set
#         transform=None # Use default ToTensor transform
#     )
#     eval_dataloader = DataLoader(
#         eval_dataset,
#         batch_size=dataset_config.get('test_batch_size', 1), # Typically batch size 1 for evaluation
#         shuffle=False, # No need to shuffle evaluation data
#         num_workers=dataset_config.get('num_workers', 4),
#         pin_memory=True and device.type == 'cuda'
#     )
#     logger.info(f"Eval DataLoader created with {len(eval_dataloader)} batches.")


#     # --- Model Setup ---
#     # Define the main network (CGNet)
#     try:
#          # Use the imported define_network
#          model = define_network(model_config).to(device)
#          logger.info("Successfully defined CGNet model.")
#     except Exception as e:
#          logger.error(f"Error defining CGNet model: {e}")
#          # Re-raise the error to stop execution if model definition fails
#          raise


#     logger.info(f"CGNet Model:\n{model}")


#     # Define and load the pre-trained Autoencoder for perceptual loss
#     ae_checkpoint_path = autoencoder_config['model_path']
#     if not os.path.exists(ae_checkpoint_path):
#         logger.error(f"Pre-trained AE checkpoint not found at {ae_checkpoint_path}. Perceptual loss will not work correctly.")
#         logger.error("Please train the Autoencoder first using scripts/train_ae.py.")
#         raise FileNotFoundError(f"Pre-trained AE checkpoint not found: {ae_checkpoint_path}")

#     try:
#          # Use the imported define_autoencoder
#          # NOTE: This import is from models.autoencoder, ensure define_autoencoder is exposed there.
#          # If models.autoencoder has a similar dynamic loading __init__.py, it might need adjustment
#          # similar to what we discussed for models.archs.
#          # from models.autoencoder import define_autoencoder # Moved import to top
#          ae_model = define_autoencoder(autoencoder_config).to(device)
#          logger.info("Successfully defined Autoencoder model for perceptual loss.")
#     except Exception as e:
#          logger.error(f"Error defining Autoencoder model: {e}")
#          raise


#     # Load AE weights and freeze the model
#     logger.info(f"Loading AE weights from {ae_checkpoint_path} and freezing the model.")
#     try:
#         ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device)
#         ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
#         for param in ae_model.parameters():
#             param.requires_grad = False # Freeze AE weights
#         ae_model.eval() # Set AE to evaluation mode
#         logger.info("AE model weights loaded and model frozen.")
#     except Exception as e:
#         logger.error(f"Error loading or freezing AE model from {ae_checkpoint_path}: {e}")
#         logger.error("Perceptual loss might not function as expected.")
#         raise e


#     # --- Loss, Optimizer, Scheduler ---
#     # Define the combined loss function
#     try:
#         # Use the imported CombinedLoss
#         # Pass all loss arguments from config, including initial_learnable_weights
#         criterion = CombinedLoss(
#             pixel_loss_type=loss_config.get('args', {}).get('pixel_loss_type', 'mae'),
#             pixel_loss_weight=loss_config.get('args', {}).get('pixel_loss_weight', 1.0),
#             ssim_loss_weight=loss_config.get('args', {}).get('ssim_loss_weight', 1.0),
#             feature_loss_weight=loss_config.get('args', {}).get('feature_loss_weight', 1.0),
#             gradient_loss_weight=loss_config.get('args', {}).get('gradient_loss_weight', 0.0), # Pass gradient weight
#             learnable_weights=loss_config.get('args', {}).get('learnable_weights', False),
#             initial_learnable_weights=loss_config.get('args', {}).get('initial_learnable_weights', None) # Pass initial weights
#         ).to(device)
#         logger.info("Successfully defined CombinedLoss.")
#     except Exception as e:
#          logger.error(f"Error defining CombinedLoss: {e}")
#          raise

#     # Attach criterion to model for easier access in save_checkpoint (optional but convenient)
#     # REMOVED: This line is likely causing the duplicate parameter warning.
#     # model.criterion = criterion


#     # Collect parameters to optimize
#     # Use separate parameter groups for model parameters and loss weights parameters
#     # This avoids the duplicate parameter warning.
#     param_groups = [{'params': model.parameters()}]
#     loss_weights_params = []

#     if hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
#        hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
#         logger.info("Adding learnable loss weights to optimizer parameters.")
#         loss_weights_params = list(criterion.loss_weights_module.parameters())
#         # Add loss weights parameters as a separate parameter group
#         # You might want to use a different learning rate for loss weights (e.g., smaller)
#         # Here, we use the same initial LR from optimizer_config['args']['lr']
#         param_groups.append({'params': loss_weights_params, 'lr': optimizer_config['args']['lr']})

#         # Log initial learnable weights if the method exists
#         if hasattr(criterion.loss_weights_module, 'get_lambdas'):
#              initial_lambdas = criterion.loss_weights_module.get_lambdas()
#              lambda_names = ['pixel', 'ssim', 'feature']
#              if hasattr(criterion.loss_weights_module, 'log_lambda_grad'): # Check if gradient lambda exists
#                   lambda_names.append('grad')
#              log_message = "Initial learnable lambdas: "
#              for name, val in zip(lambda_names, initial_lambdas):
#                   log_message += f"{name}={val.item():.4f}, "
#              logger.info(log_message.rstrip(', '))


#     # Initialize the optimizer with the parameter groups
#     optimizer = torch.optim.__dict__[optimizer_config['type']](
#         param_groups, **optimizer_config['args'])
#     logger.info(f"Optimizer: {optimizer}")


#     scheduler = None
#     if scheduler_config:
#         # Ensure scheduler type exists in torch.optim.lr_scheduler
#         if not hasattr(torch.optim.lr_scheduler, scheduler_config['type']):
#             raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
#         scheduler = torch.optim.lr_scheduler.__dict__[scheduler_config['type']](
#             optimizer, **scheduler_config['args'])
#         logger.info(f"Scheduler: {scheduler}")


#     # --- Resume Training (Optional) ---
#     start_epoch = 0
#     resume_checkpoint_path = training_config.get('resume_checkpoint')
#     if resume_checkpoint_path:
#         # If resume_checkpoint_path is relative, make it relative to the run_dir
#         if not os.path.isabs(resume_checkpoint_path):
#              resume_checkpoint_path = os.path.join(run_dir, resume_checkpoint_path)
#         # Load checkpoint for the main model, optimizer, scheduler, and criterion
#         start_epoch = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, criterion)


#     # --- Training Loop ---
#     logger.info("Starting CGNet training...")
#     # Track the best evaluation metric for saving the 'best' checkpoint
#     eval_metrics_list = evaluation_config.get('metrics', ["psnr"]) # Default to PSNR if not specified
#     if not eval_metrics_list:
#          logger.warning("No evaluation metrics specified for CGNet. Cannot track 'best' model.")
#          primary_eval_metric_name = None
#          best_eval_metric = float('-inf') # Default to negative infinity if no metric to track (higher is better assumption)
#     else:
#         primary_eval_metric_name = eval_metrics_list[0]
#         # Assuming PSNR/SSIM where higher is better. Adjust if using a loss or different metric.
#         best_eval_metric = float('-inf')
#         if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
#              best_eval_metric = float('inf')


#     for epoch in range(start_epoch, training_config['epochs']):
#         model.train() # Set main model to training mode
#         # AE model remains in eval mode and frozen
#         ae_model.eval()

#         total_loss = 0
#         total_loss_dict = {} # To accumulate individual losses for logging
#         epoch_start_time = time.time()

#         logger.info(f"Epoch {epoch}/{training_config['epochs']} starting...")

#         # Use tqdm for progress bar in console if available (optional)
#         # try:
#         #     from tqdm import tqdm
#         #     dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch}")
#         # except ImportError:
#         #     dataloader_iter = train_dataloader
#         dataloader_iter = train_dataloader # Use standard iterator


#         for batch_idx, (inputs, targets) in enumerate(dataloader_iter):
#             # --- Removed Extensive DEBUG Logging for Batch Details ---
#             # logger.debug(f"Batch {batch_idx}: Inputs shape {inputs.shape}, Targets shape {targets.shape}")
#             # logger.debug(f"Inputs device: {inputs.device}, Targets device: {targets.device}")

#             inputs, targets = inputs.to(device), targets.to(device)

#             # --- Removed Extensive DEBUG Logging for Batch Details on Device ---
#             # logger.debug(f"Batch {batch_idx}: Inputs on device shape {inputs.shape}, Targets on device shape {targets.shape}")
#             # logger.debug(f"Inputs on device: {inputs.device}, Targets on device: {targets.device}")


#             optimizer.zero_grad()
#             # logger.debug(f"Batch {batch_idx}: Optimizer gradients zeroed.") # Removed DEBUG log

#             # Forward pass through the main model (CGNet)
#             # logger.debug(f"Batch {batch_idx}: Starting model forward pass.") # Removed DEBUG log
#             outputs = model(inputs)
#             # logger.debug(f"Batch {batch_idx}: Model forward pass finished. Outputs shape: {outputs.shape}") # Removed DEBUG log
#             # logger.debug(f"Batch {batch_idx}: Outputs device: {outputs.device}") # Removed DEBUG log


#             # Get AE features (detach to prevent gradients flowing back to AE)
#             # AE model is already in eval mode and frozen
#             with torch.no_grad():
#                  # logger.debug(f"Batch {batch_idx}: Starting AE feature extraction (no_grad).") # Removed DEBUG log
#                  ae_features_pred = ae_model.get_features(outputs).detach()
#                  ae_features_gt = ae_model.get_features(targets).detach()
#                  # logger.debug(f"Batch {batch_idx}: AE feature extraction finished.") # Removed DEBUG log
#                  # logger.debug(f"Batch {batch_idx}: AE features Pred shape: {ae_features_pred.shape}, AE features GT shape: {ae_features_gt.shape}") # Removed DEBUG log
#                  # logger.debug(f"Batch {batch_idx}: AE features Pred device: {ae_features_pred.device}, AE features GT device: {ae_features_gt.device}") # Removed DEBUG log


#             # Calculate loss using the combined criterion
#             # logger.debug(f"Batch {batch_idx}: Starting loss calculation.") # Removed DEBUG log
#             # The criterion returns total loss and a dict of individual losses
#             total_batch_loss, batch_loss_dict = criterion(outputs, targets, ae_features_pred, ae_features_gt)
#             # logger.debug(f"Batch {batch_idx}: Loss calculation finished. Total loss: {total_batch_loss.item():.4f}") # Removed DEBUG log
#             # logger.debug(f"Batch {batch_idx}: Individual batch losses: {batch_loss_dict}") # Removed DEBUG log


#             # Backpropagation and optimization
#             # logger.debug(f"Batch {batch_idx}: Starting backward pass.") # Removed DEBUG log
#             total_batch_loss.backward()
#             # logger.debug(f"Batch {batch_idx}: Backward pass finished. Starting optimizer step.") # Removed DEBUG log
#             optimizer.step()
#             # logger.debug(f"Batch {batch_idx}: Optimizer step finished.") # Removed DEBUG log


#             # Accumulate total loss and individual losses for the epoch
#             total_loss += total_batch_loss.item()
#             for key, value in batch_loss_dict.items():
#                  # Ensure value is a scalar tensor before calling .item()
#                  # Also, exclude lambda values from this sum as they are weights, not losses
#                  if isinstance(value, torch.Tensor) and value.numel() == 1 and not key.startswith('lambda_'):
#                       if key not in total_loss_dict:
#                            total_loss_dict[key] = 0.0
#                       total_loss_dict[key] += value.item()


#             # Log training progress
#             if (batch_idx + 1) % training_config.get('log_interval', 50) == 0:
#                 avg_loss = total_loss / (batch_idx + 1)
#                 # Log individual batch losses and average epoch loss so far
#                 log_message = f"Epoch [{epoch}/{training_config['epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Total Loss: {total_batch_loss.item():.4f}, Avg Epoch Loss: {avg_loss:.4f}"
#                 # Add individual batch losses to the log message
#                 for key, value in batch_loss_dict.items():
#                      # Check if value is a scalar tensor and not the total_loss key itself
#                      if key != 'total_loss' and isinstance(value, torch.Tensor) and value.numel() == 1:
#                           log_message += f", {key}: {value.item():.4f}"
#                 logger.info(log_message)

#                 # Log to TensorBoard
#                 global_step = epoch * len(train_dataloader) + batch_idx
#                 writer.add_scalar('Train/batch_total_loss', total_batch_loss.item(), global_step)
#                 writer.add_scalar('Train/avg_epoch_loss', avg_loss, global_step)
#                 writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)
#                 # Log individual batch losses and learnable weights to TensorBoard
#                 for key, value in batch_loss_dict.items():
#                      if isinstance(value, torch.Tensor) and value.numel() == 1:
#                           writer.add_scalar(f'Train/batch_{key}', value.item(), global_step)


#         # End of Epoch
#         avg_epoch_loss = total_loss / len(train_dataloader)
#         avg_epoch_loss_dict = {key: total / len(train_dataloader) for key, total in total_loss_dict.items()}
#         epoch_elapsed_time = time.time() - epoch_start_time
#         logger.info(f"Epoch {epoch} finished. Avg Total Loss: {avg_epoch_loss:.4f}. Epoch time: {epoch_elapsed_time:.2f}s")
#         # Log average individual epoch losses
#         for key, avg_val in avg_epoch_loss_dict.items():
#              if key != 'total_loss':
#                   logger.info(f"  Avg Epoch {key}: {avg_val:.4f}")

#         # Log epoch loss and learnable weights to TensorBoard
#         writer.add_scalar('Train/epoch_total_loss', avg_epoch_loss, epoch)
#         for key, avg_val in avg_epoch_loss_dict.items():
#              if key != 'total_loss':
#                   writer.add_scalar(f'Train/epoch_{key}', avg_val, epoch)

#         # Log learnable weights at the end of the epoch if applicable
#         if hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
#            hasattr(criterion, 'loss_weights_module') and hasattr(criterion.loss_weights_module, 'get_lambdas'):
#              lambdas = criterion.loss_weights_module.get_lambdas()
#              lambda_names = ['pixel', 'ssim', 'feature']
#              if hasattr(criterion.loss_weights_module, 'log_lambda_grad'): # Check if gradient lambda exists
#                   lambda_names.append('grad')
#              log_message = "Learned lambdas at epoch end: "
#              for name, val in zip(lambda_names, lambdas):
#                   log_message += f"{name}={val.item():.4f}, "
#                   writer.add_scalar(f'Train/epoch_lambda_{name}', val.item(), epoch)
#              logger.info(log_message.rstrip(', '))


#         # Step the scheduler
#         if scheduler:
#             scheduler.step()
#             logger.info(f"Scheduler stepped. New LR: {optimizer.param_groups[0]['lr']:.6f}")


#         # Save checkpoint periodically
#         save_interval = training_config.get('save_interval', 10)
#         if save_interval > 0 and (epoch + 1) % save_interval == 0:
#              # Pass the average epoch loss when saving checkpoint
#              # Pass the criterion instance to save its state dict if learnable weights are used
#              save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name, criterion=criterion)

#         # Evaluate on validation set periodically
#         eval_interval = training_config.get('eval_interval', 5) # Default to 5 if not specified
#         if eval_interval > 0 and (epoch + 1) % eval_interval == 0 and len(eval_dataloader) > 0:
#             # Use the imported evaluate_model
#             logger.info(f"Starting evaluation for epoch {epoch}...")
#             # evaluate_model function does not need to know about learnable weights,
#             # it just needs the model and data to calculate metrics.
#             avg_eval_metrics = evaluate_model(model, eval_dataloader, device, logger, epoch, writer, evaluation_config.get('metrics', ["psnr", "ssim"]))

#             # Check if this is the best model based on the primary evaluation metric
#             if primary_eval_metric_name and primary_eval_metric_name in avg_eval_metrics:
#                  current_metric_value = avg_eval_metrics[primary_eval_metric_name]
#                  # Assuming PSNR/SSIM where higher is better. Adjust if using a loss or different metric.
#                  is_best = current_metric_value > best_eval_metric
#                  if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
#                       is_best = current_metric_value < best_eval_metric

#                  if is_best:
#                       best_eval_metric = current_metric_value
#                       logger.info(f"New best CGNet model found based on {primary_eval_metric_name}: {best_eval_metric:.4f}. Saving checkpoint.")
#                       # Save the best checkpoint
#                       # Pass the criterion instance to save its state dict if learnable weights are used
#                       save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name, criterion=criterion, is_best=True)
#                  else:
#                       logger.info(f"Evaluation metric {primary_eval_metric_name} ({current_metric_value:.4f}) is not better than the current best ({best_eval_metric:.4f}).")

#             else:
#                  logger.warning(f"Primary evaluation metric '{primary_eval_metric_name}' not found in evaluation results. Cannot track best model.")


#     logger.info("CGNet training finished.")
#     writer.close() # Close the TensorBoard writer


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train CGNet model.')
#     parser.add_argument('--config', type=str, default='configs/default_config.yaml',
#                         help='Path to the training configuration file.')
#     args = parser.parse_args()

#     # Ensure the config file exists
#     if not os.path.exists(args.config):
#         logger.error(f"Config file not found at {args.config}")
#         raise FileNotFoundError(f"Config file not found: {args.config}")

#     main(args.config)





# scripts/train.py

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
from datasets.ct_denoise_dataset import CTDenoiseDataset # Assuming this is correct

# --- Import actual modules ---
# Uncommenting these lines to attempt importing the real functions/classes.
# If any of these imports fail, an ImportError will be raised.
# Added specific try...except blocks for better error reporting.

# Import logging utility first
try:
    from utils.logging_utils import setup_logger
    # Setup a basic logger for initial messages before main is fully configured
    # This logger will be reconfigured in main.
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # Basic config for console output before main is called
        # Set initial level to WARNING or INFO, will be set to INFO for file logging in main
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Successfully imported and configured basic logging.")
except ImportError as e:
    # Fallback to basic logging if setup_logger cannot be imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Fallback to INFO
    logger = logging.getLogger(__name__) # Get logger again after basic config
    logger.error(f"ImportError: Could not import setup_logger from utils.logging_utils: {e}")
    logger.error("Using basic logging configuration at INFO level.")
except Exception as e:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Fallback to INFO
     logger = logging.getLogger(__name__)
     logger.error(f"Error configuring logging: {e}")
     logger.error("Using basic logging configuration at INFO level.")


# Now import other modules with specific error logging.
try:
    from models.archs import define_network
    logger.debug("Successfully imported define_network from models.archs.")
except ImportError as e:
    logger.error(f"ImportError: Could not import define_network from models.archs: {e}")
    logger.error("Please check models/archs/__init__.py and your model architecture files.")
    raise # Re-raise the error as the model is essential
except Exception as e:
    logger.error(f"Error importing define_network: {e}")
    raise # Re-raise the error

try:
    from models.autoencoder import define_autoencoder
    logger.debug("Successfully imported define_autoencoder from models.autoencoder.")
except ImportError as e:
    logger.error(f"ImportError: Could not import define_autoencoder from models.autoencoder: {e}")
    logger.error("Please check models/autoencoder/__init__.py and your AE architecture files.")
    raise # Re-raise the error as the autoencoder is essential
except Exception as e:
    logger.error(f"Error importing define_autoencoder: {e}")
    raise # Re-raise the error

try:
    from losses.combined_loss import CombinedLoss
    logger.debug("Successfully imported CombinedLoss from losses.combined_loss.")
except ImportError as e:
    logger.error(f"ImportError: Could not import CombinedLoss from losses.combined_loss: {e}")
    logger.error("Please check losses/combined_loss.py and its dependencies (like learnable_weights.py or pytorch-msssim).")
    raise # Re-raise the error as the loss is essential
except Exception as e:
    logger.error(f"Error importing CombinedLoss: {e}")
    raise # Re-raise the error

# Add specific error logging for the eval_utils import
try:
    from utils.eval_utils import evaluate_model
    logger.debug("Successfully imported evaluate_model from utils.eval_utils.")
except ImportError as e:
    logger.error(f"ImportError: Could not import evaluate_model from utils.eval_utils: {e}")
    logger.error("This is likely causing the 'Using placeholder evaluate_model' message if it appears.")
    logger.error("Please check utils/eval_utils.py and its dependencies (like utils/metrics.py or utils/logging_utils.py).")
    logger.error("Verify file paths and project structure relative to where you are running train.py.")
    raise # Re-raise the error to stop execution with a clear message
except Exception as e:
     logger.error(f"Error importing evaluate_model: {e}")
     logger.error("This is likely causing the 'Using placeholder evaluate_model' message if it appears.")
     raise # Re-raise the error

# Import metrics functions (used by evaluate_model)
try:
    from utils.metrics import calculate_psnr, calculate_ssim
    logger.debug("Successfully imported metrics functions from utils.metrics.")
except ImportError as e:
    logger.error(f"ImportError: Could not import metrics functions from utils.metrics: {e}")
    logger.error("Please check utils/metrics.py and its dependencies (like scikit-image or pytorch-msssim).")
    raise # Re-raise the error if metrics cannot be imported
except Exception as e:
    logger.error(f"Error importing metrics functions: {e}")
    raise # Re-raise the error


# --- REMOVED PLACEHOLDER DEFINITIONS ---
# The placeholder functions for define_network, define_autoencoder, CombinedLoss,
# and evaluate_model have been removed.
# If the imports above fail, the script will now raise an ImportError immediately.


def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True # Can slow down training
            # torch.backends.cudnn.benchmark = False    # Can slow down training


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, criterion=None):
    """
    Loads a model checkpoint and returns the epoch to resume from.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load the state dict into.
        optimizer (Optimizer, optional): The optimizer to load the state dict into.
        scheduler (Scheduler, optional): The scheduler to load the state dict into.
        criterion (nn.Module, optional): The criterion (CombinedLoss) to load the state dict into
                                         if it uses learnable weights.

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


    # If using learnable weights, load the criterion's state dict
    # Check if criterion is provided and if it's using learnable weights
    if criterion is not None and hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
       hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
         try:
             if 'criterion_state_dict' in checkpoint:
                  criterion.load_state_dict(checkpoint['criterion_state_dict'])
                  logger.info("Criterion state dict loaded.")
             else:
                  logger.warning("Criterion state dict not found in checkpoint. Learnable weights will start from initial values.")
         except Exception as e:
              logger.warning(f"Could not load criterion state dict from {checkpoint_path}: {e}. Learnable weights will start from initial values.")


    start_epoch = checkpoint.get('epoch', 0) + 1 # Start from the next epoch
    logger.info(f"Resuming training from epoch {start_epoch}")
    return start_epoch

def save_checkpoint(epoch, model, optimizer, scheduler, loss, checkpoint_dir, experiment_name, criterion=None, is_best=False):
    """
    Saves a model checkpoint.

    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer to save.
        scheduler (Scheduler, optional): The scheduler to save.
        loss (float): The current loss value (e.e., average epoch loss).
        checkpoint_dir (str): The directory to save checkpoints.
        experiment_name (str): The name of the experiment.
        criterion (nn.Module, optional): The criterion (CombinedLoss) to save the state dict for
                                         if it uses learnable weights.
        is_best (bool): Whether this is the best model so far (saves an additional 'best' checkpoint).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
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

    # If using learnable weights, save the criterion's state dict
    # Check if criterion is provided and if it's using learnable weights
    if criterion is not None and hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
       hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
         state['criterion_state_dict'] = criterion.state_dict()
         logger.debug("Saving criterion state dict (including learnable weights).")


    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        # Optionally save a 'latest' checkpoint symlink or copy
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


# FIX: Modify main to accept the resume_checkpoint_arg
def main(config_path, resume_checkpoint_arg=None):
    """Main training function for the CGNet model."""

    # --- Setup ---
    # Load config first to get experiment name and output directory
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the relevant sections of the config
    model_config = config['model']
    autoencoder_config = config['autoencoder']
    dataset_config = config['dataset']
    loss_config = config['loss']
    optimizer_config = config['optimizer']
    scheduler_config = config.get('scheduler') # Optional
    training_config = config['training']
    evaluation_config = config['evaluation']

    # Create experiment directory BEFORE setting up file logger
    experiment_name = training_config.get('experiment_name', 'cgnet_train_run')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine the run directory. If resuming, try to use the directory from the checkpoint path.
    # If not resuming, create a new timestamped directory.
    run_dir = None
    if resume_checkpoint_arg:
        # If resuming, the run_dir is the parent directory of the checkpoint
        run_dir = os.path.dirname(resume_checkpoint_arg)
        if not os.path.exists(run_dir):
             # Fallback to creating a new directory if the resume path parent doesn't exist
             logger.warning(f"Resume checkpoint directory not found: {run_dir}. Creating a new run directory.")
             run_dir = os.path.join(training_config.get('output_dir', 'experiments'), f'{experiment_name}_{timestamp}')
             os.makedirs(run_dir, exist_ok=True)
        else:
             logger.info(f"Resuming training from directory: {run_dir}")
    else:
        # If not resuming, create a new timestamped directory
        run_dir = os.path.join(training_config.get('output_dir', 'experiments'), f'{experiment_name}_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Starting new training run in directory: {run_dir}")


    # Setup logging - use the utility function - NOW SAFE TO CALL
    log_file = os.path.join(run_dir, 'train.log')
    # Use the imported setup_logger to configure the file logger for 'train_logger'
    # This will add a file handler to the logger obtained by getLogger('train_logger')
    # The global logger obtained by getLogger(__name__) will also be used for console output.
    # We can get the 'train_logger' instance here to ensure we are using the one with the file handler.
    train_logger = setup_logger('train_logger', log_file=log_file, level=logging.INFO) # Set train_logger to INFO

    # Now use train_logger for logging within main
    train_logger.info(f"Starting CGNet training experiment: {experiment_name}")
    train_logger.info(f"Run directory: {run_dir}")
    train_logger.info(f"Config:\n{yaml.dump(config, indent=4)}") # Log the full config


    # Save the config file used for this run into the experiment directory
    try:
        # Save the config *after* determining the run_dir (whether new or resumed)
        shutil.copyfile(config_path, os.path.join(run_dir, 'config.yaml'))
    except shutil.SameFileError:
        # This happens if config_path is already inside run_dir (e.g., resuming)
        pass
    except Exception as e:
        train_logger.warning(f"Could not copy config file to run directory: {e}")


    # Setup TensorBoard SummaryWriter
    tensorboard_log_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    train_logger.info(f"TensorBoard logs saving to: {tensorboard_log_dir}")


    # Device setup
    device = torch.device(training_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    train_logger.info(f"Using device: {device}")


    # --- Data Loading ---
    # Load training dataset and dataloader
    train_dataset = CTDenoiseDataset(
        root=dataset_config['args']['root'],
        mode='train',
        transform=None # Use default ToTensor transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataset_config.get('train_batch_size', 8),
        shuffle=True,
        num_workers=dataset_config.get('num_workers', 4),
        pin_memory=True and device.type == 'cuda'
    )
    train_logger.info(f"Train DataLoader created with {len(train_dataloader)} batches.")


    # Load evaluation dataset and dataloader (using test split)
    eval_dataset = CTDenoiseDataset(
        root=dataset_config['args']['root'],
        mode='test', # Evaluate on the test set
        transform=None # Use default ToTensor transform
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=dataset_config.get('test_batch_size', 1), # Typically batch size 1 for evaluation
        shuffle=False, # No need to shuffle evaluation data
        num_workers=dataset_config.get('num_workers', 4),
        pin_memory=True and device.type == 'cuda'
    )
    train_logger.info(f"Eval DataLoader created with {len(eval_dataloader)} batches.")


    # --- Model Setup ---
    # Define the main network (CGNet)
    try:
         # Use the imported define_network
         model = define_network(model_config).to(device)
         train_logger.info("Successfully defined CGNet model.")
    except Exception as e:
         train_logger.error(f"Error defining CGNet model: {e}")
         # Re-raise the error to stop execution if model definition fails
         raise


    train_logger.info(f"CGNet Model:\n{model}")


    # Define and load the pre-trained Autoencoder for perceptual loss
    ae_checkpoint_path = autoencoder_config['model_path']
    if not os.path.exists(ae_checkpoint_path):
        train_logger.error(f"Pre-trained AE checkpoint not found at {ae_checkpoint_path}. Perceptual loss will not work correctly.")
        train_logger.error("Please train the Autoencoder first using scripts/train_ae.py.")
        raise FileNotFoundError(f"Pre-trained AE checkpoint not found: {ae_checkpoint_path}")

    try:
         # Use the imported define_autoencoder
         # NOTE: This import is from models.autoencoder, ensure define_autoencoder is exposed there.
         # If models.autoencoder has a similar dynamic loading __init__.py, it might need adjustment
         # similar to what we discussed for models.archs.
         # from models.autoencoder import define_autoencoder # Moved import to top
         ae_model = define_autoencoder(autoencoder_config).to(device)
         train_logger.info("Successfully defined Autoencoder model for perceptual loss.")
    except Exception as e:
         train_logger.error(f"Error defining Autoencoder model: {e}")
         raise


    # Load AE weights and freeze the model
    train_logger.info(f"Loading AE weights from {ae_checkpoint_path} and freezing the model.")
    try:
        ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device)
        ae_model.load_state_dict(ae_checkpoint['model_state_dict'])
        for param in ae_model.parameters():
            param.requires_grad = False # Freeze AE weights
        ae_model.eval() # Set AE to evaluation mode
        train_logger.info("AE model weights loaded and model frozen.")
    except Exception as e:
        train_logger.error(f"Error loading or freezing AE model from {ae_checkpoint_path}: {e}")
        train_logger.error("Perceptual loss might not function as expected.")
        raise e


    # --- Loss, Optimizer, Scheduler ---
    # Define the combined loss function
    try:
        # Use the imported CombinedLoss
        # Pass all loss arguments from config, including initial_learnable_weights
        criterion = CombinedLoss(
            pixel_loss_type=loss_config.get('args', {}).get('pixel_loss_type', 'mae'),
            pixel_loss_weight=loss_config.get('args', {}).get('pixel_loss_weight', 1.0),
            ssim_loss_weight=loss_config.get('args', {}).get('ssim_loss_weight', 1.0),
            feature_loss_weight=loss_config.get('args', {}).get('feature_loss_weight', 1.0),
            gradient_loss_weight=loss_config.get('args', {}).get('gradient_loss_weight', 0.0), # Pass gradient weight
            learnable_weights=loss_config.get('args', {}).get('learnable_weights', False),
            initial_learnable_weights=loss_config.get('args', {}).get('initial_learnable_weights', None) # Pass initial weights
        ).to(device)
        train_logger.info("Successfully defined CombinedLoss.")
    except Exception as e:
         train_logger.error(f"Error defining CombinedLoss: {e}")
         raise

    # Attach criterion to model for easier access in save_checkpoint (optional but convenient)
    # REMOVED: This line is likely causing the duplicate parameter warning.
    # model.criterion = criterion


    # Collect parameters to optimize
    # Use separate parameter groups for model parameters and loss weights parameters
    # This avoids the duplicate parameter warning.
    param_groups = [{'params': model.parameters()}]
    loss_weights_params = []

    if hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
       hasattr(criterion, 'loss_weights_module') and criterion.loss_weights_module is not None:
        train_logger.info("Adding learnable loss weights to optimizer parameters.")
        loss_weights_params = list(criterion.loss_weights_module.parameters())
        # Add loss weights parameters as a separate parameter group
        # You might want to use a different learning rate for loss weights (e.g., smaller)
        # Here, we use the same initial LR from optimizer_config['args']['lr']
        param_groups.append({'params': loss_weights_params, 'lr': optimizer_config['args']['lr']}) # <-- This line sets the LR for lambdas

        # Log initial learnable weights if the method exists
        if hasattr(criterion.loss_weights_module, 'get_lambdas'):
             initial_lambdas = criterion.loss_weights_module.get_lambdas()
             lambda_names = ['pixel', 'ssim', 'feature']
             if hasattr(criterion.loss_weights_module, 'log_lambda_grad'): # Check if gradient lambda exists
                  lambda_names.append('grad')
             log_message = "Initial learnable lambdas: "
             for name, val in zip(lambda_names, initial_lambdas):
                  log_message += f"{name}={val.item():.4f}, "
             train_logger.info(log_message.rstrip(', '))


    # Initialize the optimizer with the parameter groups
    optimizer = torch.optim.__dict__[optimizer_config['type']](
        param_groups, **optimizer_config['args'])
    train_logger.info(f"Optimizer: {optimizer}")


    scheduler = None
    if scheduler_config:
        # Ensure scheduler type exists in torch.optim.lr_scheduler
        if not hasattr(torch.optim.lr_scheduler, scheduler_config['type']):
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler_config['type']](
            optimizer, **scheduler_config['args'])
        train_logger.info(f"Scheduler: {scheduler}")


    # --- Resume Training (Optional) ---
    start_epoch = 0
    resume_checkpoint_path = training_config.get('resume_checkpoint') # Get resume path from the loaded config

    if resume_checkpoint_path:
        # If resume_checkpoint_path is relative, make it relative to the run_dir
        # NOTE: run_dir is created *after* loading the checkpoint if starting fresh.
        # If resuming, we need to determine the run_dir from the checkpoint path or config.
        # A simpler approach is to assume the resume_checkpoint_path is either absolute
        # or relative to the directory where train.py is run, or relative to the output_dir.
        # Let's assume it's either absolute or relative to the current working directory for now.
        # If you need relative to run_dir, you'd need to extract run_dir from the checkpoint metadata or config.

        # For now, let's just use the provided resume_checkpoint_path directly.
        # If it's a path like experiments/..., it will be relative to where you run the script.

        # Load checkpoint for the main model, optimizer, scheduler, and criterion
        # Pass the criterion to load its state dict if learnable weights are used
        start_epoch = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, criterion)


    # --- Training Loop ---
    train_logger.info("Starting CGNet training...")
    # Track the best evaluation metric for saving the 'best' checkpoint
    eval_metrics_list = evaluation_config.get('metrics', ["psnr"]) # Default to PSNR if not specified
    if not eval_metrics_list:
         train_logger.warning("No evaluation metrics specified for CGNet. Cannot track 'best' model.")
         primary_eval_metric_name = None
         best_eval_metric = float('-inf') # Default to negative infinity if no metric to track (higher is better assumption)
    else:
        primary_eval_metric_name = eval_metrics_list[0]
        # Assuming PSNR/SSIM where higher is better. Adjust if using a loss or different metric.
        best_eval_metric = float('-inf')
        if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
             best_eval_metric = float('inf')


    for epoch in range(start_epoch, training_config['epochs']):
        model.train() # Set main model to training mode
        # AE model remains in eval mode and frozen
        ae_model.eval()

        total_loss = 0
        total_loss_dict = {} # To accumulate individual losses for logging
        epoch_start_time = time.time()

        train_logger.info(f"Epoch {epoch}/{training_config['epochs']} starting...")

        # Use tqdm for progress bar in console if available (optional)
        # try:
        #     from tqdm import tqdm
        #     dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        # except ImportError:
        #     dataloader_iter = train_dataloader
        dataloader_iter = train_dataloader # Use standard iterator


        for batch_idx, (inputs, targets) in enumerate(dataloader_iter):
            # --- Removed Extensive DEBUG Logging for Batch Details ---
            # train_logger.debug(f"Batch {batch_idx}: Inputs shape {inputs.shape}, Targets shape {targets.shape}")
            # train_logger.debug(f"Inputs device: {inputs.device}, Targets device: {targets.device}")

            inputs, targets = inputs.to(device), targets.to(device)

            # --- Removed Extensive DEBUG Logging for Batch Details on Device ---
            # train_logger.debug(f"Batch {batch_idx}: Inputs on device shape {inputs.shape}, Targets on device shape {targets.shape}")
            # train_logger.debug(f"Inputs on device: {inputs.device}, Targets on device: {inputs.device}") # Corrected typo

            optimizer.zero_grad()
            # train_logger.debug(f"Batch {batch_idx}: Optimizer gradients zeroed.") # Removed DEBUG log

            # Forward pass through the main model (CGNet)
            # train_logger.debug(f"Batch {batch_idx}: Starting model forward pass.") # Removed DEBUG log
            outputs = model(inputs)
            # train_logger.debug(f"Batch {batch_idx}: Model forward pass finished. Outputs shape: {outputs.shape}") # Removed DEBUG log
            # train_logger.debug(f"Batch {batch_idx}: Outputs device: {outputs.device}") # Removed DEBUG log


            # Get AE features (detach to prevent gradients flowing back to AE)
            # AE model is already in eval mode and frozen
            with torch.no_grad():
                 # train_logger.debug(f"Batch {batch_idx}: Starting AE feature extraction (no_grad).") # Removed DEBUG log
                 ae_features_pred = ae_model.get_features(outputs).detach()
                 ae_features_gt = ae_model.get_features(targets).detach()
                 # train_logger.debug(f"Batch {batch_idx}: AE feature extraction finished.") # Removed DEBUG log
                 # train_logger.debug(f"Batch {batch_idx}: AE features Pred shape: {ae_features_pred.shape}, AE features GT shape: {ae_features_gt.shape}") # Removed DEBUG log
                 # train_logger.debug(f"Batch {batch_idx}: AE features Pred device: {ae_features_pred.device}, AE features GT device: {ae_features_gt.device}") # Removed DEBUG log


            # Calculate loss using the combined criterion
            # train_logger.debug(f"Batch {batch_idx}: Starting loss calculation.") # Removed DEBUG log
            # The criterion returns total loss and a dict of individual losses
            total_batch_loss, batch_loss_dict = criterion(outputs, targets, ae_features_pred, ae_features_gt)
            # train_logger.debug(f"Batch {batch_idx}: Loss calculation finished. Total loss: {total_batch_loss.item():.4f}") # Removed DEBUG log
            # train_logger.debug(f"Batch {batch_idx}: Individual batch losses: {batch_loss_dict}") # Removed DEBUG log


            # Backpropagation and optimization
            # train_logger.debug(f"Batch {batch_idx}: Starting backward pass.") # Removed DEBUG log
            total_batch_loss.backward()
            # train_logger.debug(f"Batch {batch_idx}: Backward pass finished. Starting optimizer step.") # Removed DEBUG log
            optimizer.step()
            # train_logger.debug(f"Batch {batch_idx}: Optimizer step finished.") # Removed DEBUG log


            # Accumulate total loss and individual losses for the epoch
            total_loss += total_batch_loss.item()
            for key, value in batch_loss_dict.items():
                 # Ensure value is a scalar tensor before calling .item()
                 # Also, exclude lambda values from this sum as they are weights, not losses
                 if isinstance(value, torch.Tensor) and value.numel() == 1 and not key.startswith('lambda_'):
                      if key not in total_loss_dict:
                           total_loss_dict[key] = 0.0
                      total_loss_dict[key] += value.item()


            # Log training progress
            if (batch_idx + 1) % training_config.get('log_interval', 50) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                # Log individual batch losses and average epoch loss so far
                log_message = f"Epoch [{epoch}/{training_config['epochs']}], Batch [{batch_idx+1}/{len(train_dataloader)}], Total Loss: {total_batch_loss.item():.4f}, Avg Epoch Loss: {avg_loss:.4f}"
                # Add individual batch losses to the log message
                for key, value in batch_loss_dict.items():
                     # Check if value is a scalar tensor and not the total_loss key itself
                     if key != 'total_loss' and isinstance(value, torch.Tensor) and value.numel() == 1:
                          log_message += f", {key}: {value.item():.4f}"
                train_logger.info(log_message)

                # Log to TensorBoard
                global_step = epoch * len(train_dataloader) + batch_idx
                writer.add_scalar('Train/batch_total_loss', total_batch_loss.item(), global_step)
                writer.add_scalar('Train/avg_epoch_loss', avg_loss, global_step)
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)
                # Log individual batch losses and learnable weights to TensorBoard
                for key, value in batch_loss_dict.items():
                     if isinstance(value, torch.Tensor) and value.numel() == 1:
                          writer.add_scalar(f'Train/batch_{key}', value.item(), global_step)


        # End of Epoch
        avg_epoch_loss = total_loss / len(train_dataloader)
        avg_epoch_loss_dict = {key: total / len(train_dataloader) for key, total in total_loss_dict.items()}
        epoch_elapsed_time = time.time() - epoch_start_time
        train_logger.info(f"Epoch {epoch} finished. Avg Total Loss: {avg_epoch_loss:.4f}. Epoch time: {epoch_elapsed_time:.2f}s")
        # Log average individual epoch losses
        for key, avg_val in avg_epoch_loss_dict.items():
             if key != 'total_loss':
                  train_logger.info(f"  Avg Epoch {key}: {avg_val:.4f}")

        # Log epoch loss and learnable weights to TensorBoard
        writer.add_scalar('Train/epoch_total_loss', avg_epoch_loss, epoch)
        for key, avg_val in avg_epoch_loss_dict.items():
             if key != 'total_loss':
                  writer.add_scalar(f'Train/epoch_{key}', avg_val, epoch)

        # Log learnable weights at the end of the epoch if applicable
        if hasattr(criterion, 'learnable_weights') and criterion.learnable_weights and \
           hasattr(criterion, 'loss_weights_module') and hasattr(criterion.loss_weights_module, 'get_lambdas'):
             lambdas = criterion.loss_weights_module.get_lambdas()
             lambda_names = ['pixel', 'ssim', 'feature']
             if hasattr(criterion.loss_weights_module, 'log_lambda_grad'): # Check if gradient lambda exists
                  lambda_names.append('grad')
             log_message = "Learned lambdas at epoch end: "
             for name, val in zip(lambda_names, lambdas):
                  log_message += f"{name}={val.item():.4f}, "
                  writer.add_scalar(f'Train/epoch_lambda_{name}', val.item(), epoch)
             train_logger.info(log_message.rstrip(', '))


        # Step the scheduler
        if scheduler:
            scheduler.step()
            train_logger.info(f"Scheduler stepped. New LR: {optimizer.param_groups[0]['lr']:.6f}")


        # Save checkpoint periodically
        save_interval = training_config.get('save_interval', 10)
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
             # Pass the average epoch loss when saving checkpoint
             # Pass the criterion instance to save its state dict if learnable weights are used
             save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name, criterion=criterion)

        # Evaluate on validation set periodically
        eval_interval = training_config.get('eval_interval', 5) # Default to 5 if not specified
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0 and len(eval_dataloader) > 0:
            # Use the imported evaluate_model
            train_logger.info(f"Starting evaluation for epoch {epoch}...")
            # evaluate_model function does not need to know about learnable weights,
            # it just needs the model and data to calculate metrics.
            avg_eval_metrics = evaluate_model(model, eval_dataloader, device, train_logger, epoch, writer, evaluation_config.get('metrics', ["psnr", "ssim"]))

            # Check if this is the best model based on the primary evaluation metric
            if primary_eval_metric_name and primary_eval_metric_name in avg_eval_metrics:
                 current_metric_value = avg_eval_metrics[primary_eval_metric_name]
                 # Assuming PSNR/SSIM where higher is better. Adjust if using a loss or different metric.
                 is_best = current_metric_value > best_eval_metric
                 if 'loss' in primary_eval_metric_name.lower(): # If tracking a loss, lower is better
                      is_best = current_metric_value < best_eval_metric

                 if is_best:
                      best_eval_metric = current_metric_value
                      train_logger.info(f"New best CGNet model found based on {primary_eval_metric_name}: {best_eval_metric:.4f}. Saving checkpoint.")
                      # Save the best checkpoint
                      # Pass the criterion instance to save its state dict if learnable weights are used
                      save_checkpoint(epoch, model, optimizer, scheduler, avg_epoch_loss, run_dir, experiment_name, criterion=criterion, is_best=True)
                 else:
                      train_logger.info(f"Evaluation metric {primary_eval_metric_name} ({current_metric_value:.4f}) is not better than the current best ({best_eval_metric:.4f}).")

            else:
                 train_logger.warning(f"Primary evaluation metric '{primary_eval_metric_name}' not found in evaluation results. Cannot track best model.")


    train_logger.info("CGNet training finished.")
    writer.close() # Close the TensorBoard writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGNet model.')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the training configuration file.')
    # Add the --resume_checkpoint argument to the parser
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to resume training from.')
    args = parser.parse_args()

    # Ensure the config file exists
    if not os.path.exists(args.config):
        # Use the basic logger defined at the top before main is called
        logger.error(f"Config file not found at {args.config}")
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Call main with the config file path and the parsed resume_checkpoint argument
    main(args.config, args.resume_checkpoint) # Pass config path and resume arg to main

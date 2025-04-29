# scripts/preprocess_data.py

import os
import glob
import pydicom
import numpy as np
from PIL import Image
import argparse
import shutil
import logging
import random # Import random for potential shuffling

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_ima_to_png(input_path, output_path):
    """
    Converts a single .IMA (DICOM) file to a 3-channel PNG image.
    Normalizes pixel values to the range [0, 1] and converts to uint8 [0, 255] for PNG saving.
    Duplicates the single channel data across 3 channels (RGB).
    """
    try:
        ds = pydicom.dcmread(input_path)
        # Get pixel array and convert to float32
        # Handle potential multi-frame DICOMs if necessary, assuming single frame for now
        if hasattr(ds, 'PixelData'):
            img = ds.pixel_array.astype(np.float32)
        else:
            logger.error(f"No PixelData found in {input_path}. Skipping.")
            return

        # Normalize image to the range [0, 1]
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val - min_val < 1e-8: # Handle case of constant image
            logger.warning(f"Image {input_path} has constant pixel values. Normalizing to zeros.")
            img = np.zeros_like(img)
        else:
            img = (img - min_val) / (max_val - min_val)

        # Convert float [0, 1] to uint8 [0, 255] for standard PNG saving
        img_uint8 = (img * 255.0).astype(np.uint8)

        # Convert single channel to 3-channel (RGB) by stacking
        img_rgb_uint8 = np.stack([img_uint8] * 3, axis=-1)

        im = Image.fromarray(img_rgb_uint8, 'RGB')
        im.save(output_path)
        # logger.info(f"Converted {os.path.basename(input_path)} to {os.path.basename(output_path)}") # Too verbose
    except Exception as e:
        logger.error(f"Error converting {input_path}: {e}")


def process_folder(input_folder, output_folder):
    """
    Processes all .IMA files in an input folder and saves them as PNGs in an output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Use glob with case-insensitivity if needed, but .IMA is common
    ima_files = sorted(glob.glob(os.path.join(input_folder, "*.IMA")))
    if not ima_files:
        logger.warning(f"No .IMA files found in {input_folder}")
        return

    logger.info(f"Found {len(ima_files)} .IMA files in {input_folder}. Converting to PNG...")
    for i, file in enumerate(ima_files):
        base_name = os.path.basename(file)
        # Ensure output filename is safe and consistent
        out_name = os.path.splitext(base_name)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)
        convert_ima_to_png(file, out_path)
        if (i + 1) % 100 == 0: # Log progress every 100 files
            logger.info(f"Processed {i + 1}/{len(ima_files)} files in {input_folder}")
    logger.info(f"Finished converting files in {input_folder}.")


def split_dataset(full_png_dir, quarter_png_dir, output_dir, train_ratio=0.8, train_pairs=None, seed=None):
    """
    Splits paired full-dose and quarter-dose PNG images into train/test sets.

    Args:
        full_png_dir (str): Directory containing full-dose (NDCT) PNG files.
        quarter_png_dir (str): Directory containing quarter-dose (LDCT) PNG files.
        output_dir (str): Root directory to save the split dataset ('train' and 'test' subfolders will be created here).
        train_ratio (float): Ratio of data to use for training (if train_pairs is None).
        train_pairs (int): Explicit number of pairs to use for training (overrides train_ratio if not None).
        seed (int, optional): Random seed for shuffling the data before splitting. Defaults to None.
    """
    full_files = sorted(glob.glob(os.path.join(full_png_dir, "*.png")))
    quarter_files = sorted(glob.glob(os.path.join(quarter_png_dir, "*.png")))

    if not full_files or not quarter_files:
        logger.error("No PNG files found in source directories for splitting. Ensure conversion step was successful.")
        return

    # Basic check for matching filenames (assuming they correspond after sorting)
    if len(full_files) != len(quarter_files):
         logger.error(f"Mismatch in number of full-dose ({len(full_files)}) and quarter-dose ({len(quarter_files)}) images. Cannot split.")
         # You might add more sophisticated matching logic here if filenames don't directly correspond after sorting.
         raise AssertionError("Mismatch in number of full-dose and quarter-dose images.")

    num_total_pairs = len(full_files)
    logger.info(f"Found {num_total_pairs} total paired PNG images for splitting.")

    # Determine the number of training pairs
    if train_pairs is not None:
        if train_pairs > num_total_pairs:
            logger.warning(f"Requested train_pairs ({train_pairs}) is more than total pairs ({num_total_pairs}). Using all pairs for training.")
            num_train_pairs = num_total_pairs
        else:
             num_train_pairs = train_pairs
    else:
        num_train_pairs = int(num_total_pairs * train_ratio)

    num_test_pairs = num_total_pairs - num_train_pairs

    # Ensure at least one pair in test set if possible
    if num_test_pairs == 0 and num_total_pairs > 0:
         if num_train_pairs > 1:
            num_train_pairs -= 1
            num_test_pairs = 1
            logger.warning(f"Adjusted split: {num_train_pairs} train, {num_test_pairs} test to ensure test set is not empty.")
         else:
             logger.warning("Only one total pair available, cannot create a separate test set.")


    logger.info(f"Splitting dataset: {num_train_pairs} pairs for training, {num_test_pairs} pairs for testing.")

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(os.path.join(train_dir, "full"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "quarter"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "full"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "quarter"), exist_ok=True)

    # Create a list of indices and shuffle if a seed is provided
    indices = list(range(num_total_pairs))
    if seed is not None:
        logger.info(f"Using random seed {seed} for shuffling data split.")
        random.seed(seed)
        random.shuffle(indices)

    # Get indices for train and test sets
    train_indices = indices[:num_train_pairs]
    test_indices = indices[num_train_pairs:]

    logger.info("Copying files for training set...")
    for i in train_indices:
        try:
            shutil.copy(full_files[i], os.path.join(train_dir, "full", os.path.basename(full_files[i])))
            shutil.copy(quarter_files[i], os.path.join(train_dir, "quarter", os.path.basename(quarter_files[i])))
        except Exception as e:
            logger.error(f"Error copying train file index {i}: {e}")

    logger.info("Copying files for testing set...")
    for i in test_indices:
        try:
            shutil.copy(full_files[i], os.path.join(test_dir, "full", os.path.basename(full_files[i])))
            shutil.copy(quarter_files[i], os.path.join(test_dir, "quarter", os.path.basename(quarter_files[i])))
        except Exception as e:
             logger.error(f"Error copying test file index {i}: {e}")


    logger.info("Dataset splitting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IMA to PNG and split dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory containing IMA files (e.g., 'data/raw/L067'). "
                             "Should contain subfolders like 'full_1mm' and 'quarter_1mm'.")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for processed data. PNG images will be saved to "
                             "output_dir/converted_png and splits to output_dir/split.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of pairs to use for training (used if --train_pairs is not set).")
    parser.add_argument("--train_pairs", type=int, default=None,
                        help="Explicit number of pairs to use for training (overrides --train_ratio).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling data before splitting.")


    args = parser.parse_args()

    # Define input and output paths based on arguments
    full_input_dir = os.path.join(args.data_dir, "full_1mm")
    quarter_input_dir = os.path.join(args.data_dir, "quarter_1mm")

    converted_png_output_dir_full = os.path.join(args.output_dir, "converted_png", "full_1mm")
    converted_png_output_dir_quarter = os.path.join(args.output_dir, "converted_png", "quarter_1mm")

    split_output_dir = os.path.join(args.output_dir, "split")

    logger.info("Starting data preprocessing...")

    # Convert IMA files to PNG for both full and quarter dose images.
    process_folder(full_input_dir, converted_png_output_dir_full)
    process_folder(quarter_input_dir, converted_png_output_dir_quarter)

    # Split the dataset into train/test using the converted PNGs
    split_dataset(converted_png_output_dir_full, converted_png_output_dir_quarter,
                  split_output_dir, train_ratio=args.train_ratio, train_pairs=args.train_pairs, seed=args.seed)

    logger.info("Data preprocessing complete.")

# datasets/ct_denoise_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CTDenoiseDataset(Dataset):
    """
    Custom Dataset for loading paired LDCT (quarter-dose) and NDCT (full-dose) images.
    Assumes images are stored in 'root/mode/quarter' and 'root/mode/full' directories.
    Images are expected to be 3-channel (RGB) PNG files.
    """
    def __init__(self, root, mode="train", transform=None):
        """
        Args:
            root (str): Path to the split dataset root folder (e.g., 'data/processed/split').
                        This folder should contain subfolders 'train' and 'test'.
            mode (str): Dataset mode, either "train" or "test".
            transform (callable, optional): Optional transform to be applied to the images.
                                            Defaults to converting to PyTorch Tensor.
        """
        self.mode = mode
        # Construct paths to the quarter-dose (input) and full-dose (target) image directories
        self.quarter_dir = os.path.join(root, mode, "quarter")
        self.full_dir = os.path.join(root, mode, "full")

        if not os.path.exists(self.quarter_dir):
            logger.error(f"Quarter-dose image directory not found: {self.quarter_dir}")
            self.quarter_images = []
            self.full_images = []
            return
        if not os.path.exists(self.full_dir):
             logger.error(f"Full-dose image directory not found: {self.full_dir}")
             self.quarter_images = []
             self.full_images = []
             return


        # List and sort image files in both directories
        # Sorting ensures that the pairs correspond correctly based on filenames
        self.quarter_images = sorted([os.path.join(self.quarter_dir, f) for f in os.listdir(self.quarter_dir) if f.endswith('.png')])
        self.full_images = sorted([os.path.join(self.full_dir, f) for f in os.listdir(self.full_dir) if f.endswith('.png')])

        # Assert that the number of input and target images is the same
        if len(self.full_images) != len(self.quarter_images):
             logger.error(f"Mismatch in number of images in {self.quarter_dir} ({len(self.quarter_images)}) and {self.full_dir} ({len(self.full_images)}).")
             # You might want to add more robust error handling or pair matching here
             raise AssertionError(f"Mismatch in number of images for mode '{mode}'.")

        if len(self.full_images) == 0:
             logger.warning(f"No PNG images found in {self.quarter_dir} or {self.full_dir} for mode '{mode}'.")


        # Define the transformation. Default is ToTensor.
        # ToTensor converts PIL Image (H, W, C) in range [0, 255] to
        # PyTorch Tensor (C, H, W) in range [0.0, 1.0].
        self.transform = transform if transform is not None else transforms.ToTensor()

        logger.info(f"Initialized CTDenoiseDataset in '{mode}' mode with {len(self)} image pairs.")


    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.full_images)

    def __getitem__(self, idx):
        """
        Retrieves an image pair (quarter-dose, full-dose) at the given index.

        Args:
            idx (int): Index of the image pair.

        Returns:
            tuple: (quarter_img, full_img) - quarter-dose image (input) and full-dose image (target).
                   Both are PyTorch Tensors after applying the transform, in the range [0.0, 1.0].
        """
        # Get the file paths for the quarter-dose and full-dose images
        quarter_path = self.quarter_images[idx]
        full_path = self.full_images[idx]

        try:
            # Open the images using PIL
            # .convert("RGB") ensures 3 channels, even if the source PNG was grayscale
            quarter_img = Image.open(quarter_path).convert("RGB")
            full_img = Image.open(full_path).convert("RGB")

            # Apply the specified transform
            if self.transform:
                quarter_img = self.transform(quarter_img)
                full_img = self.transform(full_img)

            # Return the input (quarter-dose) and target (full-dose) images
            return quarter_img, full_img

        except Exception as e:
            logger.error(f"Error loading image pair at index {idx}: {quarter_path}, {full_path}. Error: {e}")
            # Depending on requirements, you might return None or raise the error
            # Returning dummy data or skipping might be needed in a real training loop
            # For now, let's re-raise the error to indicate a data loading issue
            raise e


# Example Usage (for testing the dataset)
if __name__ == '__main__':
    # This block requires you to have some dummy data in the specified structure
    # Example: Create dummy directories and files for testing
    dummy_root = "dummy_data_split"
    dummy_train_q_dir = os.path.join(dummy_root, "train", "quarter")
    dummy_train_f_dir = os.path.join(dummy_root, "train", "full")
    dummy_test_q_dir = os.path.join(dummy_root, "test", "quarter")
    dummy_test_f_dir = os.path.join(dummy_root, "test", "full")

    os.makedirs(dummy_train_q_dir, exist_ok=True)
    os.makedirs(dummy_train_f_dir, exist_ok=True)
    os.makedirs(dummy_test_q_dir, exist_ok=True)
    os.makedirs(dummy_test_f_dir, exist_ok=True)

    # Create dummy PNG files (e.g., 10 train pairs, 5 test pairs)
    dummy_img = Image.new('RGB', (128, 128), color = 'red')
    num_dummy_train = 10
    num_dummy_test = 5
    for i in range(num_dummy_train):
        dummy_img.save(os.path.join(dummy_train_q_dir, f"img_{i:03d}.png"))
        dummy_img.save(os.path.join(dummy_train_f_dir, f"img_{i:03d}.png"))
    for i in range(num_dummy_test):
        dummy_img.save(os.path.join(dummy_test_q_dir, f"test_img_{i:03d}.png"))
        dummy_img.save(os.path.join(dummy_test_f_dir, f"test_img_{i:03d}.png"))

    print(f"Created dummy data in {dummy_root} ({num_dummy_train} train, {num_dummy_test} test pairs)")

    try:
        # Instantiate the training dataset
        train_dataset = CTDenoiseDataset(root="data/split", mode="train") # <-- Change root
        print(f"Train dataset size: {len(train_dataset)}")
        if len(train_dataset) > 0:
            # Get a sample item
            input_img, target_img = train_dataset[0]
            print(f"Sample train item shape: Input {input_img.shape}, Target {target_img.shape}")
            print(f"Sample train item dtype: Input {input_img.dtype}, Target {target_img.dtype}")
            print(f"Sample train item value range: Input [{input_img.min():.4f}, {input_img.max():.4f}], Target [{target_img.min():.4f}, {target_img.max():.4f}]")


        # Instantiate the testing dataset
        test_dataset = CTDenoiseDataset(root="data/split", mode="test")   # <-- Change root
        print(f"Test dataset size: {len(test_dataset)}")
        if len(test_dataset) > 0:
            # Get a sample item
            input_img, target_img = test_dataset[0]
            print(f"Sample test item shape: Input {input_img.shape}, Target {target_img.shape}")
            print(f"Sample test item dtype: Input {input_img.dtype}, Target {target_img.dtype}")
            print(f"Sample test item value range: Input [{input_img.min():.4f}, {input_img.max():.4f}], Target [{target_img.min():.4f}, {target_img.max():.4f}]")


        # Test with a custom transform (e.g., resizing)
        custom_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        train_dataset_resized = CTDenoiseDataset(root=dummy_root, mode="train", transform=custom_transform)
        print(f"Train dataset size with resize transform: {len(train_dataset_resized)}")
        if len(train_dataset_resized) > 0:
            input_img_resized, target_img_resized = train_dataset_resized[0]
            print(f"Sample train item with resize transform shape: Input {input_img_resized.shape}, Target {target_img_resized.shape}")


    except Exception as e:
        logger.error(f"An error occurred during dataset test: {e}")
        print("Please ensure dummy data is created correctly or run with actual data.")

    finally:
        # Clean up dummy data
        # import shutil # Uncomment if you want to automatically remove dummy data
        # if os.path.exists(dummy_root):
        #     shutil.rmtree(dummy_root)
        print(f"Dummy data left in {dummy_root}. Please remove manually if needed.")


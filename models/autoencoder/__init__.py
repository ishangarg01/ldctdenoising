# models/autoencoder/__init__.py

# This file handles the dynamic import and instantiation of Autoencoder architectures
# defined within the 'autoencoder' directory.
import importlib
from os import path as osp
import sys
import os # Using os.listdir for directory scanning
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to scan directory for files ending with a suffix
def scandir(dir_path, suffix):
    """Scan a directory for files ending with a specific suffix."""
    files = []
    if not osp.isdir(dir_path):
        logger.warning(f"Directory not found for scanning: {dir_path}")
        return files
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isfile(full_path) and entry.endswith(suffix):
            files.append(entry)
    return files

# automatically scan and import autoencoder modules
# scan all the files under the 'autoencoder' folder and collect files ending with
# '.py' (excluding __init__.py). These files are expected to contain AE classes.
ae_folder = osp.dirname(osp.abspath(__file__)) # Get the directory of this file
ae_filenames = [
    osp.splitext(v)[0] for v in scandir(ae_folder, '.py') if not v.startswith('__init__')
]

# import all the found autoencoder modules dynamically
_ae_modules = []
logger.info(f"Scanning directory for Autoencoder modules: {ae_folder}")
if not ae_filenames:
     logger.warning("No Autoencoder modules (*.py excluding __init__.py) found in the 'models/autoencoder/' directory.")

for file_name in ae_filenames:
    module_name = f'models.autoencoder.{file_name}' # Construct the full module name
    try:
        # Import the module using its relative path within the project structure
        module = importlib.import_module(module_name)
        _ae_modules.append(module)
        logger.info(f"Successfully imported Autoencoder module: {module_name}")
    except ImportError as e:
        logger.warning(f"Could not import Autoencoder module {module_name}. It might be missing dependencies or have syntax errors. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while importing {module_name}: {e}")

if not _ae_modules:
    logger.error("No Autoencoder modules were found or successfully imported in the 'models/autoencoder/' directory. Cannot define Autoencoders.")


def dynamic_instantiation_ae(modules, cls_type, opt):
    """Dynamically instantiate an Autoencoder class from a list of imported modules.

    Searches through the provided list of AE modules for a class with the name
    `cls_type` and instantiates the first one found using the keyword
    arguments provided in `opt`.

    Args:
        modules (list[importlib modules]): List of imported AE modules to search within.
        cls_type (str): The name of the AE class to instantiate.
        opt (dict): Dictionary of keyword arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified AE class.

    Raises:
        ValueError: If the class type is not found in any of the provided modules.
        TypeError: If `opt` is not a dictionary.
    """
    if not isinstance(opt, dict):
        raise TypeError(f"Options for dynamic AE instantiation must be a dictionary, but got {type(opt)}")

    for module in modules:
        # Try to get the class attribute from the current module
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            # Found the class, instantiate it with provided options and return
            logger.info(f"Instantiating AE class '{cls_type}' from module '{module.__name__}'")
            return cls_(**opt)

    # If the loop finishes without finding the class in any module
    raise ValueError(f'Autoencoder class type "{cls_type}" is not found in any of the loaded AE modules.')


def define_autoencoder(opt):
    """Defines and instantiates an Autoencoder based on configuration options.

    This is the entry point for creating AE models from configuration.

    Args:
        opt (dict): Configuration dictionary for the Autoencoder. Must contain an 'arch' key
                    with a dictionary specifying the AE architecture. The 'arch' dictionary
                    must contain a 'type' key (e.g., 'SimpleConvAE') and optionally
                    an 'args' key with a dictionary of initialization arguments.

    Returns:
        nn.Module: The instantiated Autoencoder model.

    Raises:
        ValueError: If the 'arch' or 'type' key is missing in the configuration or if the
                    specified AE type cannot be instantiated.
    """
    if not isinstance(opt, dict):
         raise TypeError(f"Autoencoder configuration must be a dictionary, but got {type(opt)}")
    if 'arch' not in opt or not isinstance(opt['arch'], dict):
        raise ValueError("Autoencoder configuration dictionary must contain an 'arch' key with a dictionary.")
    if 'type' not in opt['arch']:
        raise ValueError("Autoencoder architecture configuration dictionary must contain a 'type' key specifying the class name.")

    ae_arch_opt = opt['arch']
    ae_type = ae_arch_opt['type']
    # Get arguments for the AE constructor, default to empty dict if 'args' is missing
    ae_args = ae_arch_opt.get('args', {})

    logger.info(f"Defining Autoencoder of type: {ae_type} with args: {ae_args}")

    # Dynamically instantiate the AE class using the loaded modules
    ae_model = dynamic_instantiation_ae(_ae_modules, ae_type, ae_args)

    return ae_model

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This example requires at least one AE .py file (excluding __init__.py)
    # with a class matching the dummy config type.
    # For example, if you create simple_conv_ae.py with class SimpleConvAE:
    dummy_config = {
        'arch': {
            'type': 'SimpleConvAE', # This should match a class name in a file like simple_conv_ae.py
            'args': {
                'in_channels': 3,
                'base_channels': 16,
                'num_encoder_layers': 3
            }
        },
        # Other AE config might be here, but define_autoencoder only uses 'arch'
    }
    print("--- Testing dynamic Autoencoder instantiation ---")
    try:
        # Define the AE using the dummy config
        dummy_ae = define_autoencoder(dummy_config)
        print("\nSuccessfully defined a dummy Autoencoder:")
        print(dummy_ae)

        # Test with a non-existent AE type
        dummy_config_invalid = {
            'arch': {
                'type': 'NonExistentAE',
                'args': {}
            }
        }
        print("\nAttempting to define a non-existent Autoencoder:")
        try:
            define_autoencoder(dummy_config_invalid)
        except ValueError as e:
            print(f"Caught expected error: {e}")

    except ValueError as e:
         print(f"\nCould not test dynamic AE instantiation: {e}")
         print("Please ensure you have at least one .py file (excluding __init__.py) in models/autoencoder/ with a class matching the dummy config type.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example AE usage: {e}")

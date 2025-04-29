# models/archs/__init__.py
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# This file handles the dynamic import and instantiation of network architectures
# defined within the 'archs' directory.
# ------------------------------------------------------------------------
import importlib
from os import path as osp
import sys
import os
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Setup a basic console logger initially if not already configured
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


# --- Fix: Explicitly import the define function for CGNet and expose it ---
# This approach is more robust than relying solely on dynamic instantiation
# through a list of potentially failed imports, and ensures define_network
# is directly accessible if the module loads.
try:
    # Assuming your CGNet architecture is in CGNet_arch.py and has a function define_CGNet
    # Use relative import from the same package
    from .CGNet_arch import define_CGNet

    # Assign the imported function to the name 'define_network' in this module's namespace.
    # This makes it accessible when importing from the models.archs package (e.g., from models.archs import define_network).
    define_network = define_CGNet
    logger.info("Successfully imported define_CGNet and assigned it to define_network.")

    # If you had multiple architectures, you might instead build a dictionary here
    # and modify the define_network function below to look up the correct model type.
    # Example:
    # model_registry = {
    #     'CGNet': define_CGNet,
    #     # Add other models here: 'NAFNet': define_NAFNet, # Assuming you have a NAFNet_arch.py with define_NAFNet
    # }
    # logger.info(f"Registered models: {list(model_registry.keys())}")

except ImportError as e:
    # Log the specific import error for debugging
    logger.error(f"Failed to import define_CGNet from CGNet_arch.py: {e}")
    # Define a placeholder define_network function that raises a clear error if called.
    # This ensures that if the actual model cannot be loaded, you get an informative error
    # instead of silently using a dummy network.
    def define_network(opt):
        logger.error("Actual define_network function could not be loaded due to import errors.")
        raise NotImplementedError(
            "Actual define_network function could not be loaded. "
            "Please check models/archs/CGNet_arch.py and its dependencies (like arch_util.py) "
            "for syntax errors or missing dependencies within those files."
        )
    logger.warning("Using a placeholder define_network that will raise NotImplementedError if called.")

except Exception as e:
     # Catch any other unexpected errors during the import process
     logger.error(f"An unexpected error occurred during define_network setup in __init__.py: {e}")
     def define_network(opt):
          logger.error("Actual define_network function could not be loaded due to an unexpected error during setup.")
          raise RuntimeError(f"Error during define_network setup: {e}")
     logger.warning("Using a placeholder define_network that will raise a RuntimeError if called.")


# --- Optional: Keep dynamic scanning if you plan to add more architectures ---
# The following code scans for other _arch.py files. You might keep this
# if you want a registry of models, but the primary define_network for CGNet
# is handled above. If you use a registry (as shown in the commented example above),
# you'll need to modify the `define_network` placeholder function above to use the registry.

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'. These files are expected to contain network architecture classes.
# arch_folder = osp.dirname(osp.abspath(__file__)) # Already defined above
# arch_filenames = [
#     osp.splitext(v)[0] for v in scandir(arch_folder, '_arch.py')
# ]

# Get the package name for relative imports (if needed for dynamic registry population)
# current_package = '.'.join(__name__.split('.')[:-1]) # This would be 'models' if this is models.archs.__init__

# _arch_modules = []
# if not arch_filenames:
#      logger.warning("No files ending with '_arch.py' found in the 'models/archs/' directory.")

# Example of populating a registry (if needed for multiple models)
# for file_name in arch_filenames:
#     # Skip the file we already explicitly handled if needed
#     if file_name == 'CGNet_arch':
#         continue # Skip CGNet_arch as its define function is handled above

#     module_name = f'.{file_name}'
#     try:
#         # Import the module using its relative path within the package
#         # module = importlib.import_module(module_name, package=current_package + '.archs')
#         # _arch_modules.append(module)
#         # logger.info(f"Successfully imported architecture module: {module_name}")

#         # Example: Find define_* functions in other modules and add to registry
#         # for attribute_name in dir(module):
#         #      attribute = getattr(module, attribute_name)
#         #      if callable(attribute) and attribute_name.startswith('define_'):
#         #           model_type = attribute_name.replace('define_', '')
#         #           # Add to your registry dictionary defined earlier
#         #           # model_registry[model_type] = attribute
#         #           logger.info(f"Registered '{model_type}' from {file_name}.")


#     except ImportError as e:
#         logger.error(f"Failed to import architecture module {file_name}: {e}")
#     except Exception as e:
#          logger.error(f"An error occurred during processing module {file_name}: {e}")

# If using a registry, your main define_network function would use it:
# def define_network(opt):
#     model_type = opt.get('arch', {}).get('type')
#     if model_type in model_registry:
#         return model_registry[model_type](opt)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")


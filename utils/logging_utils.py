# utils/logging_utils.py

import logging
import sys
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO, stream=True):
    """
    Sets up a logger with console and optionally file handlers.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, no file handler is added.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        stream (bool): Whether to add a console (stream) handler. Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(name)
    # Prevent adding handlers multiple times if the logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False # Prevent logs from being passed to the root logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add console handler if requested
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Add file handler if log_file path is provided
    if log_file is not None:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Example Usage (for testing the logger setup)
if __name__ == '__main__':
    print("--- Testing logging_utils.py ---")

    # Test setting up a logger with console output only
    console_logger = setup_logger('console_only_logger', stream=True, log_file=None, level=logging.DEBUG)
    console_logger.debug("This is a debug message (console only)")
    console_logger.info("This is an info message (console only)")
    console_logger.warning("This is a warning message (console only)")
    console_logger.error("This is an error message (console only)")

    # Test setting up a logger with console and file output
    # Create a dummy log file path in a temporary directory
    log_dir = "temp_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_logger = setup_logger('file_and_console_logger', log_file=log_file_path, stream=True, level=logging.INFO)
    file_logger.info(f"Logging to console and file: {log_file_path}")
    file_logger.info("This is an info message (file and console)")
    file_logger.warning("This is a warning message (file and console)")

    # Test getting the same logger instance
    same_logger = setup_logger('file_and_console_logger')
    if same_logger is file_logger:
        print("\nSuccessfully retrieved the existing logger instance.")
        same_logger.info("This message should go to the same logger.")
    else:
        print("\nFailed to retrieve the existing logger instance.")

    # Clean up dummy directory
    # import shutil # Uncomment if you want to automatically remove dummy data
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    print(f"\nDummy log directory '{log_dir}' created. Please remove manually if needed.")
    print("--- Logging test complete ---")

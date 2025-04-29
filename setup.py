# setup.py

from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of the requirements.txt file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="ldct-denoising-cgnet", # Replace with your project name
    version="0.1.0", # Start with a small version number
    author="Your Name", # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="LDCT Image Denoising using CGNet with Perceptual and Multi-Loss Supervision", # Short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourrepository", # Replace with your GitHub repo URL if applicable
    packages=find_packages(), # Automatically find packages (directories with __init__.py)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: [Your License]", # Choose your license, e.g., MIT License
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8', # Specify minimum Python version
    install_requires=install_requires, # List dependencies from requirements.txt
    # Add entry points if you want to create executable scripts directly from the package
    # entry_points={
    #     'console_scripts': [
    #         'preprocess_data=scripts.preprocess_data:main',
    #         'train_ae=scripts.train_ae:main',
    #         'evaluate_ae=scripts.evaluate_ae:main',
    #         'train_cgnet=scripts.train:main', # Renamed to avoid conflict with 'train' command
    #         'test_cgnet=scripts.test:main',   # Renamed
    #         'evaluate_cgnet=scripts.evaluate:main', # Renamed
    #     ],
    # },
)

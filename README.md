LDCT Image Denoising using CGNet with Perceptual and Multi-Loss Supervision
This project implements a deep learning pipeline for denoising Low-Dose CT (LDCT) images using a CGNet architecture, supervised by a composite loss function incorporating pixel-wise, SSIM, and perceptual components.

Project Structure
project_root/
├── configs/                 # Configuration files for experiments
├── data/                    # Raw and processed data
├── models/                  # Model architectures (CGNet, Autoencoder)
├── datasets/                # Custom dataset classes
├── losses/                  # Custom loss functions
├── utils/                   # Utility functions
├── scripts/                 # Executable scripts (train, test, preprocess, evaluate)
├── experiments/             # Results of training runs (checkpoints, logs)
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── setup.py                 # Optional: Project packaging

Setup
Clone the repository:

git clone <repository_url>
cd project_root

Create the directory structure:

mkdir -p {configs,data/{raw,processed/{converted_png/{full_1mm,quarter_1mm},split/{train/{full,quarter},test/{full,quarter}}}},models/{archs,autoencoder},datasets,losses,utils,scripts,experiments}

Install dependencies:

pip install -r requirements.txt

Place your raw data:
Place your original .IMA files in data/raw/. You should create subdirectories within data/raw for different datasets or patients, e.g., data/raw/L067/full_1mm/ and data/raw/L067/quarter_1mm/.

Preprocess the data:
Run the preprocessing script to convert .IMA to 3-channel .png and split the dataset. Adjust --data_dir and --output_dir as needed.

python scripts/preprocess_data.py --data_dir data/raw/L067 --output_dir data/processed --train_ratio 0.8 # Example usage

Train the Autoencoder:
The perceptual loss requires a pre-trained autoencoder. You need to train the AE separately using scripts/train_ae.py on your NDCT images.

python scripts/train_ae.py --config configs/ae_config.yaml # You will need to create ae_config.yaml

Ensure the trained AE checkpoint is saved to the path specified in your main CGNet training config (configs/default_config.yaml or your experiment-specific config).

Evaluate the Autoencoder (Recommended):
Before training CGNet, evaluate the trained AE to ensure it reconstructs well.

python scripts/evaluate_ae.py --config configs/ae_config.yaml --checkpoint path/to/your/trained_ae.pth

Usage
Configure your experiment:
Edit or create a YAML file in the configs/ directory (e.g., configs/my_experiment.yaml) to define model parameters, dataset paths, training settings, etc. You can start by copying configs/default_config.yaml. Make sure the autoencoder.model_path points to your trained AE checkpoint.

Train the CGNet model:
Run the training script, specifying your configuration file.

python scripts/train.py --config configs/my_experiment.yaml

Training progress, logs, and checkpoints will be saved in the experiments/ directory.

Monitor training:
Use TensorBoard to visualize training metrics.

tensorboard --logdir experiments/

Test the model:
Run the testing script to perform inference on the test set.

python scripts/test.py --config configs/my_experiment.yaml --checkpoint experiments/your_run_timestamp/your_checkpoint.pth --output_dir results/your_run_timestamp

Evaluate the model:
Run the evaluation script to compute metrics (PSNR, SSIM) on a dataset using a trained model.

python scripts/evaluate.py --config configs/my_experiment.yaml --checkpoint experiments/your_run_timestamp/your_checkpoint.pth

Extending the Project
Add new CGNet architectures: Create a new Python file for your model architecture in models/archs/ ending with _arch.py. Ensure it inherits from torch.nn.Module. Update the configs/ file with the new model.type.

Add new Autoencoder architectures: Create a new Python file for your AE class in models/autoencoder/. Update the configs/ file with the new autoencoder.arch.type. Ensure it has a method or attribute to access the encoder part for feature extraction (e.g., .encoder or .get_features()).

Add new datasets: Create a new Python file for your dataset class in datasets/ inheriting from torch.utils.data.Dataset. Update the configs/ file with the new dataset.type.

Add new loss functions: Implement new loss functions in the losses/ directory. Modify CombinedLoss or create a new loss combination strategy.

Add new metrics: Implement new metric calculation functions in utils/metrics.py.

License
[Specify your license here]
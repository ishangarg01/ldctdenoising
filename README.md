# LDCT Image Denoising using CGNet with Perceptual and Multi-Loss Supervision

This project implements a deep learning pipeline for denoising Low-Dose CT (LDCT) images using a CGNet architecture. The training process is guided by a composite loss function combining pixel-wise, SSIM, and perceptual components.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd project_root
    ```

2.  **Ensure this directory structure:**
    ```bash
    mkdir -p \
        configs \
        data/{raw,processed/{converted_png/{full_1mm,quarter_1mm},split/{train/{full,quarter},test/{full,quarter}}}} \
        models/{archs,autoencoder} \
        datasets \
        losses \
        utils \
        scripts \
        experiments
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

### Place your raw data:

Store your original `.IMA` files in `data/raw/`. For example:

data/raw/L067/full_1mm/
data/raw/L067/quarter_1mm/
### Preprocess the data:

Convert `.IMA` files to 3-channel `.png` format and split into training and test sets:

```
python scripts/preprocess_data.py \
    --data_dir data/raw/L067 \
    --output_dir data/processed \
    --train_ratio 0.8
```

## Autoencoder (AE)

### Train the Autoencoder:

The perceptual loss requires a trained autoencoder. Train it separately on NDCT images:

```
python scripts/train_ae.py --config configs/ae_config.yaml
Ensure configs/ae_config.yaml is correctly configured with dataset and model settings.Evaluate the Autoencoder (Recommended):Evaluate the AE to ensure good reconstruction quality:python scripts/evaluate_ae.py \
    --config configs/ae_config.yaml \
    --checkpoint path/to/your/trained_ae.pth
```

## CGNet Training and Evaluation

### Configure your experiment:

Create or edit a configuration file in `configs/`, for example:

`configs/my_experiment.yaml`

This should include model parameters, paths, training settings, and point to the AE checkpoint.

### Train the CGNet:

```
python scripts/train.py --config configs/my_experiment.yaml
# Logs and checkpoints will be stored in experiments/.Monitor training with TensorBoard:tensorboard --logdir experiments/
#Test the model:

python scripts/test.py \
    --config configs/my_experiment.yaml \
    --checkpoint experiments/<run_id>/<checkpoint>.pth \
    --output_dir results/<run_id>
```

### Evaluate model performance:

```
python scripts/evaluate.py \
    --config configs/my_experiment.yaml \
    --checkpoint experiments/<run_id>/<checkpoint>.pth
```

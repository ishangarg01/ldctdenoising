# models/autoencoder/enhanced_conv_ae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedConvAE(nn.Module):
    """
    An enhanced convolutional Autoencoder architecture with more layers and channels.
    Designed to learn richer features for perceptual loss.
    """
    def __init__(self, in_channels=3, base_channels=32, num_encoder_layers=4):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            base_channels (int): Number of channels in the first convolutional layer.
                                 Channel counts will increase significantly with depth.
                                 Increased default from 16 to 32.
            num_encoder_layers (int): Number of downsampling steps in the encoder.
                                      The decoder will have the same number of upsampling steps.
                                      Increased default from 3 to 4.
        """
        super(EnhancedConvAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_encoder_layers = num_encoder_layers

        # --- Encoder ---
        encoder_layers = []
        current_channels = in_channels

        # Initial convolution before the main downsampling stages
        encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(current_channels, base_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        current_channels = base_channels

        # Downsampling stages
        for i in range(num_encoder_layers):
            out_channels = base_channels * (2 ** (i + 1)) # Double channels at each stage
            # Each stage: Conv -> ReLU -> Conv -> ReLU -> Downsample Conv -> ReLU
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    # Downsampling convolution
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        # The output of the encoder is the feature map used for perceptual loss
        self.feature_channels = current_channels # Store the number of channels in the feature map

        logger.info(f"EnhancedConvAE Encoder created with {num_encoder_layers} downsampling stages.")
        logger.info(f"Encoder output channels: {self.feature_channels}")


        # --- Decoder ---
        decoder_layers = []
        # Start from the feature channels and work backwards through upsampling stages
        current_channels = self.feature_channels

        # Upsampling stages (mirroring encoder)
        for i in range(num_encoder_layers - 1, -1, -1):
            # Channels after upsampling
            out_channels = base_channels * (2 ** i) if i > 0 else base_channels
            # Each stage: Upsample ConvTranspose -> ReLU -> Conv -> ReLU -> Conv -> ReLU
            decoder_layers.append(
                nn.Sequential(
                    # Upsampling convolution
                    nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_channels

        # Final convolution layer to map back to input channels
        decoder_layers.append(
             nn.Conv2d(current_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )

        # Add a Sigmoid activation at the very end to constrain output to [0, 1]
        self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid())

        logger.info(f"EnhancedConvAE Decoder created with {num_encoder_layers} upsampling stages.")


    def forward(self, x):
        """
        Forward pass of the Autoencoder.

        Args:
            x (torch.Tensor): Input tensor (e.g., NDCT image) of shape (N, C, H, W).

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (N, C, H, W).
        """
        # Pass through encoder
        encoded = self.encoder(x)
        # Pass through decoder
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_features(self, x):
        """
        Extracts feature maps from the encoder.
        This method is used during CGNet training for perceptual loss.

        Args:
            x (torch.Tensor): Input tensor (e.g., CGNet output or ground truth)
                             of shape (N, C, H, W).

        Returns:
            torch.Tensor: Feature map tensor from the last encoder layer.
        """
        # Pass through encoder and return the output
        return self.encoder(x)

# Example Usage (for testing the AE architecture)
if __name__ == '__main__':
    # Create a dummy instance of the AE
    in_channels = 3
    base_channels = 32 # Use the enhanced base channels
    num_layers = 4 # Use the enhanced number of layers
    print(f"Instantiating EnhancedConvAE with in_channels={in_channels}, base_channels={base_channels}, num_encoder_layers={num_layers}")
    ae_model = EnhancedConvAE(in_channels=in_channels, base_channels=base_channels, num_encoder_layers=num_layers)
    print("Model instantiated successfully.")
    print(ae_model)

    # Create dummy input data (e.g., batch size 2, 3 channels, 256x256 resolution)
    # Input spatial dimensions should be divisible by 2^num_encoder_layers (2^4=16)
    dummy_input_shape = (2, in_channels, 256, 256) # Use a larger size divisible by 16
    dummy_input = torch.randn(dummy_input_shape)
    print(f"\nTesting forward pass with dummy input shape: {dummy_input_shape}")

    try:
        # Test forward pass (reconstruction)
        reconstructed_output = ae_model(dummy_input)
        print(f"Forward pass successful. Reconstructed output shape: {reconstructed_output.shape}")
        assert reconstructed_output.shape == dummy_input_shape, "Reconstructed output shape mismatch!"
        print("Reconstructed output shape matches input shape.")
        print(f"Reconstructed output value range: [{reconstructed_output.min().item():.4f}, {reconstructed_output.max().item():.4f}]")


        # Test feature extraction
        features = ae_model.get_features(dummy_input)
        # Calculate expected feature spatial size: H / (2^num_layers), W / (2^num_layers)
        expected_feature_spatial_size = (dummy_input_shape[2] // (2 ** num_layers), dummy_input_shape[3] // (2 ** num_layers))
        # Calculate expected feature channels: base_channels * 2^num_layers
        expected_feature_channels = base_channels * (2 ** num_layers)


        print(f"\nTesting feature extraction...")
        print(f"Expected feature shape: (Batch, {expected_feature_channels}, {expected_feature_spatial_size[0]}, {expected_feature_spatial_size[1]})")
        print(f"Actual feature shape: {features.shape}")

        assert features.shape[0] == dummy_input_shape[0], "Feature batch size mismatch!"
        assert features.shape[1] == expected_feature_channels, f"Feature channel mismatch! Expected {expected_feature_channels}, got {features.shape[1]}"
        assert features.shape[2:] == expected_feature_spatial_size, "Feature spatial size mismatch!"
        print("Feature shape matches expected shape.")

    except Exception as e:
        logger.error(f"An error occurred during AE model test: {e}")
        print("Please check the model implementation and dummy input shape.")

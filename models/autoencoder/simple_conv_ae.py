# models/autoencoder/simple_conv_ae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleConvAE(nn.Module):
    """
    A simple convolutional Autoencoder architecture.
    Designed to be relatively small and trainable on a modest GPU,
    used for learning features for perceptual loss.
    """
    def __init__(self, in_channels=3, base_channels=16, num_encoder_layers=3):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            base_channels (int): Number of channels in the first convolutional layer.
                                 Channel counts will increase with depth.
            num_encoder_layers (int): Number of downsampling steps in the encoder.
                                      The decoder will have the same number of upsampling steps.
        """
        super(SimpleConvAE, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_encoder_layers = num_encoder_layers

        # --- Encoder ---
        encoder_layers = []
        current_channels = in_channels
        for i in range(num_encoder_layers):
            out_channels = base_channels * (2 ** i) # Double channels at each layer
            # Use Conv2d with stride 2 for downsampling
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        # The output of the encoder is the feature map used for perceptual loss
        self.feature_channels = current_channels # Store the number of channels in the feature map

        logger.info(f"SimpleConvAE Encoder created with {num_encoder_layers} layers.")
        # logger.debug(f"Encoder output channels: {self.feature_channels}")


        # --- Decoder ---
        decoder_layers = []
        # Start from the feature channels and work backwards
        current_channels = self.feature_channels
        for i in range(num_encoder_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** (i - 1)) if i > 0 else in_channels
            # Use ConvTranspose2d with stride 2 for upsampling
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True) if i > 0 else nn.Identity() # No ReLU on the last layer
                )
            )
            current_channels = out_channels

        # Final layer to output the reconstructed image (Sigmoid for [0, 1] range)
        decoder_layers.append(
             nn.Conv2d(current_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        # Add a Sigmoid activation at the very end to constrain output to [0, 1]
        self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid())

        logger.info(f"SimpleConvAE Decoder created with {num_encoder_layers} layers.")


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
    base_channels = 16
    num_layers = 3
    print(f"Instantiating SimpleConvAE with in_channels={in_channels}, base_channels={base_channels}, num_encoder_layers={num_layers}")
    ae_model = SimpleConvAE(in_channels=in_channels, base_channels=base_channels, num_encoder_layers=num_layers)
    print("Model instantiated successfully.")
    print(ae_model)

    # Create dummy input data (e.g., batch size 2, 3 channels, 128x128 resolution)
    # Input spatial dimensions should be divisible by 2^num_encoder_layers (2^3=8)
    dummy_input_shape = (2, in_channels, 128, 128)
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
        expected_feature_spatial_size = (dummy_input_shape[2] // (2 ** num_layers), dummy_input_shape[3] // (2 ** num_layers))
        expected_feature_channels = base_channels * (2 ** (num_layers - 1)) # Channels before the last doubling if num_layers > 0
        if num_layers > 0:
            expected_feature_channels = base_channels * (2 ** (num_layers - 1))
        else:
             expected_feature_channels = in_channels # If 0 layers, encoder is identity

        # Correct expected feature channels calculation:
        # After layer 0: base_channels * 2^0 = base_channels
        # After layer 1: base_channels * 2^1
        # After layer num_encoder_layers-1: base_channels * 2^(num_encoder_layers-1)
        expected_feature_channels = base_channels * (2 ** (num_layers - 1)) if num_layers > 0 else in_channels # Corrected logic

        print(f"Testing feature extraction...")
        print(f"Expected feature shape: (Batch, {expected_feature_channels}, {expected_feature_spatial_size[0]}, {expected_feature_spatial_size[1]})")
        print(f"Actual feature shape: {features.shape}")

        assert features.shape[0] == dummy_input_shape[0], "Feature batch size mismatch!"
        assert features.shape[1] == expected_feature_channels, f"Feature channel mismatch! Expected {expected_feature_channels}, got {features.shape[1]}"
        assert features.shape[2:] == expected_feature_spatial_size, "Feature spatial size mismatch!"
        print("Feature shape matches expected shape.")

    except Exception as e:
        logger.error(f"An error occurred during AE model test: {e}")
        print("Please check the model implementation and dummy input shape.")


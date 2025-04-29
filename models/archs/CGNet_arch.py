# models/archs/CGNet_arch.py

# ------------------------------------------------------------------------
# Modified from CascadedGaze (https://github.com/Ascend-Research/CascadedGaze)
# ------------------------------------------------------------------------
# BasicSR-PyTorch (https://github.com/XPixelGroup/BasicSR)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)
# Ensure logger has handlers if not already configured by the main script
if not logger.handlers:
    # Set default level to INFO, but scripts like train.py can override this
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Import necessary components from your local arch_util and local_arch
# Corrected import for LayerNorm2d - it is in arch_util.py, not local_arch.py
try:
    from .arch_util import default_conv, LayerNorm2d
    logger.debug("Successfully imported default_conv and LayerNorm2d from arch_util.")
except ImportError as e:
    logger.error(f"Failed to import from arch_util: {e}")
    # Define placeholders or raise error if essential
    LayerNorm2d = lambda c: nn.Identity() # Placeholder if import fails
    default_conv = lambda *args, **kwargs: nn.Identity() # Placeholder if import fails


# Note: Local_Base and AvgPool2d from local_arch are not used in this architecture.
try:
    # Attempt to import something from local_arch to confirm it's accessible
    # from .local_arch import SomeClass # Replace with an actual import from your local_arch if needed
    logger.debug("Successfully imported local_arch.")
except ImportError as e:
    logger.warning(f"Could not import from local_arch: {e}")


# --- Integrated Classes from the provided CascadedGaze code ---

class SimpleGate(nn.Module):
    """ Simple gating mechanism: splits input channels in half and multiplies. """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1) # Split channels into two halves
        return x1 * x2 # Element-wise multiplication

class depthwise_separable_conv(nn.Module):
    """ Depthwise Separable Convolution """
    def __init__(self, nin, nout, kernel_size = 3, padding = 0, stride = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        # Depthwise convolution: applies a single filter to each input channel
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=False)
        # Pointwise convolution: a 1x1 convolution to combine the output of the depthwise layer
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class GlobalContextExtractor(nn.Module):
    """ Global Context Extractor using cascaded depthwise separable convolutions. """
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4], padding=0, bias=False): # Corrected default strides based on usage in CascadedGazeBlock
        super(GlobalContextExtractor, self).__init__()
        self.c = c # Store channel count

        # Create a list of depthwise separable convolutions
        self.depthwise_separable_convs = nn.ModuleList()
        # The input to the first conv is 'c' channels, subsequent inputs are also 'c' as output channels are 'c'
        in_c = c
        out_c = c
        for kernel_size, stride in zip(kernel_sizes, strides):
             # Calculate padding for 'same' effect if kernel is odd and stride is 1 (though strides are > 1 here)
             # For strides > 1, padding is usually 0 or calculated differently depending on desired output size.
             # The original code snippet shows padding=0 in the GCE constructor, so we'll keep that.
             calculated_padding = padding # Use the provided padding

             self.depthwise_separable_convs.append(
                 depthwise_separable_conv(in_c, out_c, kernel_size, calculated_padding, stride, bias)
             )
             # The output channels of depthwise_separable_conv are 'out_c', which is 'c' for the next layer's input
             in_c = out_c


    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            list[Tensor]: List of output tensors from each depthwise separable convolution.
        """
        outputs = []
        current_x = x # Start with the input tensor
        for i, conv in enumerate(self.depthwise_separable_convs):
            # Apply GELU activation after each convolution
            current_x = F.gelu(conv(current_x))
            outputs.append(current_x) # Append the output of each conv layer
            # logger.debug(f"GCE Conv {i} output shape: {current_x.shape}") # Debug logging

        return outputs


class CascadedGazeBlock(nn.Module):
    """ Cascaded Gaze Block with Global Context Extraction and Channel Mixing. """
    def __init__(self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0):
        super().__init__()
        self.c = c # Store channel count
        self.dw_channel = c * DW_Expand # Depthwise convolution expansion factor
        self.GCE_Conv = GCE_Conv # Number of GCE convolutions (determines GCE structure)

        # First convolution (1x1 pointwise)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1,
                                padding=0, stride=1, groups=1, bias=True)
        # Second convolution (3x3 depthwise)
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
                                kernel_size=3, padding=1, stride=1, groups=self.dw_channel, # Padding 1 for 'same' spatial size
                                bias=True)

        # Global Context Extractor based on GCE_Conv parameter
        # The strides are hardcoded in the provided snippet, let's use those.
        if self.GCE_Conv == 3:
            # GCE expects 'c' input channels and outputs 'c' per layer
            self.GCE = GlobalContextExtractor(c=self.c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])
            # Project out convolution after concatenating features
            # Input channels: dw_channel (from conv2) + sum of output channels from GCE (GCE_Conv * c).
            self.project_out = nn.Conv2d(int(self.dw_channel + self.GCE_Conv * self.c), c, kernel_size=1)

            # Simplified Channel Attention operating on concatenated features
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=int(self.dw_channel + self.GCE_Conv * self.c), out_channels=int(self.dw_channel + self.GCE_Conv * self.c), kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))
        else: # Assuming GCE_Conv == 2 based on the provided code structure
            # GCE expects 'c' input channels and outputs 'c' per layer
            self.GCE = GlobalContextExtractor(c=self.c, kernel_sizes=[3, 3], strides=[2, 3]) # Use 2 kernels/strides
            # Project out convolution after concatenating features
            # Input channels: dw_channel (from conv2) + sum of output channels from GCE (GCE_Conv * c).
            self.project_out = nn.Conv2d(self.dw_channel + self.GCE_Conv * self.c, c, kernel_size=1)

            # Simplified Channel Attention operating on concatenated features
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=self.dw_channel + self.GCE_Conv * self.c, out_channels=self.dw_channel + self.GCE_Conv * self.c, kernel_size=1, padding=0, stride=1,
                        groups=1, bias=True))

        # Add a projection layer before GCE input to map dw_channel // 2 to c
        # This addresses the mismatch between `x_1 + x_2` (dw_channel // 2) and GCE's expected input (c)
        # CORRECTED: Input channels should be dw_channel // 2, as that's the size of x_1 + x_2
        self.gce_input_proj = nn.Conv2d(self.dw_channel // 2, self.c, kernel_size=1)


        # SimpleGate after the first two convolutions
        self.sg = SimpleGate()

        # Feedforward Network (FFN)
        ffn_channel = FFN_Expand * c # FFN expansion factor
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # The provided code has ffn_channel // 2 for conv5 input, implying SimpleGate is used within FFN
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Layer Normalization
        self.norm1 = LayerNorm2d(c) # Normalize input before first conv block
        self.norm2 = LayerNorm2d(c) # Normalize input to FFN

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() # Dropout after GCE and projection
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() # Dropout after FFN

        # Learnable parameters for residual connections
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp # Store input for the first residual connection
        b,c,h,w = x.shape

        # Nearest neighbor upsampling for range fusion - created dynamically in forward
        # This Upsample is used to bring the spatial size of GCE outputs back to the size before GCE
        # The size should match the spatial size of 'x' after conv2, which is the same as input size (h,w) due to padding=1 and stride=1
        upsample_layer = nn.Upsample(size=(h,w), mode='nearest')


        # First block: LayerNorm -> Conv1 (1x1) -> Conv2 (3x3 DW) -> GELU
        x = self.norm1(x)
        x = self.conv1(x) # Channels: c -> dw_channel
        x = self.conv2(x) # Channels: dw_channel -> dw_channel, spatial size maintained
        x = F.gelu(x) # Apply activation

        # Global Context Extraction and Range Fusion
        # Split x into two halves along the channel dimension for GCE input
        x_1 , x_2 = x.chunk(2, dim=1) # Each half has dw_channel // 2 channels

        # Project the sum of halves (dw_channel // 2) to 'c' channels for GCE input
        # CORRECTED: Input to gce_input_proj is x_1 + x_2, which has dw_channel // 2 channels
        gce_input = self.gce_input_proj(x_1 + x_2) # Channels: dw_channel // 2 -> c

        # Apply GCE to the projected input
        gce_outputs = self.GCE(gce_input) # GCE expects 'c' input and outputs 'c' per layer


        # Upsample GCE outputs to match the spatial size of x (after conv2)
        upsampled_gce_outputs = [upsample_layer(out) for out in gce_outputs]

        # Concatenate x (after conv2, dw_channel channels) with upsampled GCE outputs (each 'c' channels)
        # Total channels: dw_channel + GCE_Conv * c
        x = torch.cat([x] + upsampled_gce_outputs, dim=1) # Concatenate along channel dimension

        # Apply Simplified Channel Attention
        x = self.sca(x) * x

        # Project out to 'c' channels
        # Input channels: dw_channel + GCE_Conv * c
        # Output channels: c
        x = self.project_out(x) # Channels: (dw_channel + GCE_Conv * c) -> c

        # Apply dropout
        x = self.dropout1(x)

        # First residual connection
        y = inp + x * self.beta # inp has 'c' channels, x has 'c' channels

        # Second block: LayerNorm -> Conv4 (1x1 FFN expand) -> SimpleGate -> Conv5 (1x1 FFN reduce)
        x = self.conv4(self.norm2(y)) # Channels: c -> ffn_channel
        x = self.sg(x) # SimpleGate splits ffn_channel into ffn_channel // 2 and multiplies
        x = self.conv5(x) # Channels: ffn_channel // 2 -> c

        # Apply dropout
        x = self.dropout2(x)

        # Second residual connection
        return y + x * self.gamma # y has 'c' channels, x has 'c' channels


class NAFBlock0(nn.Module):
    """ NAFNet-like Block (used in the middle of CascadedGaze). """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        self.c = c # Store channel count
        dw_channel = c * DW_Expand # Depthwise convolution expansion factor

        # First convolution block: 1x1 -> 3x3 DW -> SimpleGate -> Simplified Channel Attention -> 1x1
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, # Padding 1 for 'same' spatial size
                               bias=True)
        # conv3 takes the output of SimpleGate (dw_channel // 2 channels)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention operating on the output of SimpleGate (dw_channel // 2 channels)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate after conv2
        self.sg = SimpleGate()

        # Feedforward Network (FFN)
        ffn_channel = FFN_Expand * c # FFN expansion factor
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # conv5 takes the output of SimpleGate within FFN (ffn_channel // 2 channels)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Layer Normalization
        self.norm1 = LayerNorm2d(c) # Normalize input before first conv block
        self.norm2 = LayerNorm2d(c) # Normalize input to FFN

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() # Dropout after first conv block
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() # Dropout after FFN

        # Learnable parameters for residual connections
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp # Store input for the first residual connection

        # First block: LayerNorm -> Conv1 (1x1) -> Conv2 (3x3 DW) -> SimpleGate -> SCA -> Conv3 (1x1)
        x = self.norm1(x) # Normalize input
        x = self.conv1(x) # Channels: c -> dw_channel
        x = self.conv2(x) # Channels: dw_channel -> dw_channel, spatial size maintained
        x = self.sg(x) # SimpleGate splits dw_channel into dw_channel // 2 and multiplies
        x = x * self.sca(x) # Apply channel attention to the gated output
        x = self.conv3(x) # Channels: dw_channel // 2 -> c

        # Apply dropout
        x = self.dropout1(x)

        # First residual connection
        y = inp + x * self.beta # inp has 'c' channels, x has 'c' channels

        # Second block (FFN): LayerNorm -> Conv4 (1x1 FFN expand) -> SimpleGate -> Conv5 (1x1 FFN reduce)
        x = self.conv4(self.norm2(y)) # Channels: c -> ffn_channel
        x = self.sg(x) # SimpleGate splits ffn_channel into ffn_channel // 2 and multiplies
        x = self.conv5(x) # Channels: ffn_channel // 2 -> c

        # Apply dropout
        x = self.dropout2(x)

        # Second residual connection
        return y + x * self.gamma # y has 'c' channels, x has 'c' channels


class CGNet(nn.Module): # Renamed from CascadedGaze to CGNet to match config
    """ CGNet (based on CascadedGaze architecture) for Image Denoising. """

    # Added out_channels parameter to the constructor
    def __init__(self, img_channel=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], GCE_CONVS_nums=[]):
        super().__init__()
        logger.info(f"Instantiating CGNet with img_channel={img_channel}, out_channels={out_channels}, width={width}, middle_blk_num={middle_blk_num}, enc_blk_nums={enc_blk_nums}, dec_blk_nums={dec_blk_nums}, GCE_CONVS_nums={GCE_CONVS_nums}")

        self.img_channel = img_channel # Store image channel count
        self.out_channels = out_channels # Store output channel count
        self.width = width # Store base width
        self.middle_blk_num = middle_blk_num # Store middle block count
        self.enc_blk_nums = enc_blk_nums # Store encoder block counts per stage
        self.dec_blk_nums = dec_blk_nums # Store decoder block counts per stage
        self.GCE_CONVS_nums = GCE_CONVS_nums # Store GCE conv counts per stage

        # Initial convolution layer
        # Use the 'width' parameter for the output channels of the intro layer
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=self.width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)

        # Encoder, Decoder, Middle Blocks, Upsampling, and Downsampling layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Added ModuleList for skip connection channel adjustment convolutions
        # We need a skip_conv for each encoder stage that provides a skip connection
        self.skip_convs = nn.ModuleList()


        chan = self.width # Start with base width received by the constructor
        # Build Encoder stages
        # Ensure enc_blk_nums and GCE_CONVS_nums have the same length
        assert len(enc_blk_nums) == len(GCE_CONVS_nums), "Length of enc_blk_nums and GCE_CONVS_nums must match."
        num_encoder_stages = len(enc_blk_nums)

        for i in range(num_encoder_stages):
            num_blocks = enc_blk_nums[i]
            gce_convs = GCE_CONVS_nums[i]
            self.encoders.append(
                nn.Sequential(
                    # Each encoder stage consists of CascadedGazeBlocks
                    *[CascadedGazeBlock(chan, GCE_Conv=gce_convs) for _ in range(num_blocks)] # Pass 'chan' to block
                )
            )
            # Downsampling layer after each encoder stage (except the last one)
            if i < num_encoder_stages - 1:
                # Add a 1x1 convolution to the skip connection path to adjust channels
                # The skip connection from encoder stage i (channels width * (2**i))
                # needs to map to the same number of channels as the upsampled tensor
                # in the corresponding decoder stage. The upsampled tensor in decoder
                # stage j (which corresponds to encoder stage num_decoder_stages - 1 - j)
                # has width * (2**(num_encoder_stages - 2 - j)) channels.
                # For encoder stage i, the corresponding decoder stage index is
                # j = num_decoder_stages - 1 - i = (num_encoder_stages - 1) - 1 - i = num_encoder_stages - 2 - i.
                # The upsampled tensor in decoder stage j has width * (2**(num_encoder_stages - 2 - j))
                # = width * (2**(num_encoder_stages - 2 - (num_encoder_stages - 2 - i))) = width * (2**i) channels.
                # So, the skip connection from encoder stage i (channels width * (2**i))
                # should map to width * (2**i) channels.
                # The current 'chan' in the encoder loop is width * (2**i).
                # FIX: Change output channels from chan // 2 to chan.
                self.skip_convs.append(nn.Conv2d(chan, chan, kernel_size=1)) # Keep the same number of channels


                self.downs.append(
                    nn.Conv2d(chan, 2 * chan, 2, 2) # Downsample and double channels
                )
                chan = chan * 2 # Double channel count for the next stage

        # Middle blocks (Bottleneck)
        # Input channels to middle blocks are the output channels of the last encoder stage
        # which is width * (2 ** (num_encoder_stages - 1))
        middle_in_channels = self.width * (2 ** (num_encoder_stages - 1)) # Use self.width
        self.middle_blks = \
            nn.Sequential(
                # Middle blocks use NAFBlock0
                *[NAFBlock0(middle_in_channels) for _ in range(middle_blk_num)] # Pass 'middle_in_channels' to block
            )
        chan = middle_in_channels # Channel count remains the same after middle blocks

        # Build Decoder stages
        # The number of decoder stages should be one less than encoder stages to match upsampling/downsampling factors
        num_decoder_stages = num_encoder_stages - 1
        # Ensure dec_blk_nums has enough elements for the decoder stages we are building
        assert len(dec_blk_nums) >= num_decoder_stages, "Not enough decoder block numbers specified in config."

        # Decoder stages are built in reverse order of encoder stages (excluding the last encoder stage)
        for i in range(num_decoder_stages):
            num_blocks = dec_blk_nums[i] # Use the first num_decoder_stages elements from dec_blk_nums
            # Upsampling layer before each decoder stage
            # Input channels: current_channels (output of previous decoder stage or middle blocks)
            # Output channels: current_channels * 2 (before PixelShuffle halves it)
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), # 1x1 conv to double channels
                    nn.PixelShuffle(2) # PixelShuffle with factor 2 for 2x upsampling
                )
            )
            chan = chan // 2 # Halve channel count after upsampling and for the decoder blocks
            self.decoders.append(
                nn.Sequential(
                    # Each decoder stage consists of NAFBlock0s
                    *[NAFBlock0(chan) for _ in range(num_blocks)] # Pass 'chan' to block
                )
            )

        # Final convolution layer - uses out_channels for output
        # Input channels should match the final decoder stage output channel count ('chan' after the last decoder loop iteration)
        # We determine the input channels dynamically here.
        final_decoder_channels = chan # 'chan' holds the channel count after the last decoder stage
        self.ending = nn.Conv2d(in_channels=final_decoder_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        logger.debug(f"Ending layer initialized with in_channels={final_decoder_channels}, out_channels={out_channels}")


        # Calculate the padding size needed to make input dimensions divisible by 2^num_encoder_stages
        self.padder_size = 2 ** num_encoder_stages

    def forward(self, inp):
        logger.debug(f"--- CGNet Forward Pass Start ---")
        # Get original batch size, channels, height, and width
        B, C, H, W = inp.shape
        logger.debug(f"Input shape: {inp.shape}")

        # Pad the input image if its dimensions are not divisible by padder_size
        inp_padded = self.check_image_size(inp)
        logger.debug(f"Padded input shape: {inp_padded.shape}")


        # Initial convolution
        x = self.intro(inp_padded)
        logger.debug(f"Intro output shape: {x.shape}")

        # Encoder path
        encs = [] # List to store encoder outputs for skip connections
        logger.debug(f"--- Encoder Path Start ---")
        # Iterate through encoders and downs layers
        for i in range(len(self.encoders)):
             logger.debug(f"Encoder Stage {i} input shape: {x.shape}")
             x = self.encoders[i](x) # Process through encoder blocks
             logger.debug(f"Encoder Stage {i} output shape (before down): {x.shape}")
             encs.append(x) # Store output before downsampling for skip connection
             if i < len(self.downs): # Check if there is a downsampling layer for this stage
                  x = self.downs[i](x) # Downsample spatial size and double channels
                  logger.debug(f"Encoder Stage {i} output shape (after down): {x.shape}")
        logger.debug(f"--- Encoder Path End ---")


        # Middle blocks (Bottleneck)
        logger.debug(f"--- Bottleneck Start ---")
        logger.debug(f"Bottleneck input shape: {x.shape}")
        x = self.middle_blks(x)
        logger.debug(f"Bottleneck output shape: {x.shape}")
        logger.debug(f"--- Bottleneck End ---")


        # Decoder path with skip connections
        logger.debug(f"--- Decoder Path Start ---")
        # Iterate through decoders, upsampling layers, and encoder outputs in reverse order
        # The number of decoder stages is len(self.decoders)
        num_decoder_stages = len(self.decoders)

        for i in range(num_decoder_stages):
            # Get the corresponding upsampling layer and decoder stage
            up = self.ups[i]
            decoder = self.decoders[i]
            # Get the corresponding encoder skip connection
            # The skip connection for decoder stage i comes from encoder stage num_decoder_stages - 1 - i.
            enc_skip_idx = num_decoder_stages - 1 - i
            enc_skip = encs[enc_skip_idx]

            logger.debug(f"Decoder Stage {i} (corresponds to Encoder Stage {enc_skip_idx}) input shape: {x.shape}")
            logger.debug(f"Decoder Stage {i} original skip connection shape: {enc_skip.shape}")


            x = up(x) # Upsample spatial size and halve channels
            logger.debug(f"Decoder Stage {i} output shape (after up): {x.shape}")

            # --- Apply the corresponding skip convolution for channel adjustment ---
            # The skip_convs list has `num_encoder_stages - 1` elements, indexed 0 to num_encoder_stages - 2.
            # The skip connection from encoder stage `enc_skip_idx` needs a skip_conv if `enc_skip_idx < len(self.skip_convs)`.
            # In this corrected architecture, the skip connections used in the decoder stages (0 to num_decoder_stages - 1)
            # come from encoder stages (num_decoder_stages - 1 - i), where i goes from 0 to num_decoder_stages - 1.
            # This means enc_skip_idx goes from num_decoder_stages - 1 down to 0.
            # Since num_decoder_stages = num_encoder_stages - 1, enc_skip_idx goes from num_encoder_stages - 2 down to 0.
            # These indices perfectly match the indices of self.skip_convs.
            if enc_skip_idx < len(self.skip_convs): # This check should now always pass for relevant skips
                 skip_conv = self.skip_convs[enc_skip_idx] # Get the corresponding skip convolution
                 logger.debug(f"Decoder Stage {i}: Applying skip_conv from skip_convs[{enc_skip_idx}]")

                 # Resize the encoder skip connection to match the upsampled output spatial size before applying skip_conv
                 if x.shape[-2:] != enc_skip.shape[-2:]:
                      logger.warning(f"Spatial size mismatch in decoder stage {i}: Upsampled output {x.shape[-2:]} vs Skip connection {enc_skip.shape[-2:]}. Resizing skip connection.")
                      enc_skip_resized = F.interpolate(enc_skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                      logger.debug(f"Decoder Stage {i} resized skip connection shape (spatial): {enc_skip_resized.shape}")
                 else:
                      enc_skip_resized = enc_skip


                 # Ensure the input channels to the skip_conv match the resized skip connection
                 if skip_conv.in_channels != enc_skip_resized.shape[1]:
                      logger.error(f"Skip conv input channel mismatch in decoder stage {i}: Skip conv expects {skip_conv.in_channels}, got {enc_skip_resized.shape[1]}")
                      raise RuntimeError(f"Skip conv input channel mismatch in decoder stage {i}")

                 enc_skip_adjusted = skip_conv(enc_skip_resized) # Apply the correct skip_conv
                 logger.debug(f"Decoder Stage {i} skip connection shape (after channel adjustment): {enc_skip_adjusted.shape}")

                 # Check if channel dimensions match after channel adjustment
                 # FIX: The adjusted skip connection should now have the same channels as the upsampled tensor
                 if enc_skip_adjusted.shape[1] != x.shape[1]:
                      logger.error(f"Channel mismatch after resizing and adjusting skip connection in decoder stage {i}: Adjusted skip channels {enc_skip_adjusted.shape[1]} vs Upsampled channels {x.shape[1]}")
                      raise RuntimeError(f"Channel mismatch after resizing and adjusting skip connection in decoder stage {i}")

                 x = x + enc_skip_adjusted # Add the adjusted skip connection
            else:
                 # This case would only happen if there's a skip connection from the very first encoder stage
                 # connecting to the very last decoder stage, and we didn't create a skip_conv for it.
                 # Based on the (num_encoder_stages - 1) decoder stages logic, this should not occur.
                 logger.error(f"Decoder Stage {i}: Unexpected skip connection from encoder stage {enc_skip_idx}. This requires a skip_conv.")
                 raise RuntimeError(f"Decoder Stage {i}: Missing skip convolution for encoder stage {enc_skip_idx}.")


            logger.debug(f"Decoder Stage {i} output shape (after skip): {x.shape}")
            x = decoder(x) # Process through decoder blocks
            logger.debug(f"Decoder Stage {i} output shape (after decoder): {x.shape}")

        logger.debug(f"--- Decoder Path End ---")

        # Final convolution
        logger.debug(f"Ending layer input shape: {x.shape}")
        # Ensure the input channels to the ending layer match its expected input channels
        if x.shape[1] != self.ending.in_channels:
             logger.error(f"Ending layer input channel mismatch: Ending layer expects {self.ending.in_channels}, got {x.shape[1]}")
             raise RuntimeError(f"Ending layer input channel mismatch")

        output = self.ending(x)
        logger.debug(f"Ending conv output shape: {output.shape}")


        # Add the final output to the original padded input (residual connection)
        # Note: The residual connection is added to the padded input, then cropped.
        # Ensure output and padded input have the same spatial size for residual connection
        if output.shape[-2:] != inp_padded.shape[-2:]:
             logger.error(f"Output and padded input spatial size mismatch for residual connection: Output {output.shape[-2:]} vs Padded Input {inp_padded.shape[-2:]}")
             raise RuntimeError(f"Output and padded input spatial size mismatch")

        output = output + inp_padded
        logger.debug(f"Final output shape (before crop): {output.shape}")


        # Crop the output back to the original height and width
        output = output[:, :, :H, :W]
        logger.debug(f"Final output shape (after crop): {output.shape}")
        logger.debug(f"--- CGNet Forward Pass End ---")

        return output

    def check_image_size(self, x):
        """
        Pads the input image to make its spatial dimensions divisible by self.padder_size.
        """
        _, _, h, w = x.size()
        # Calculate padding needed for height and width
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # Apply padding (left, right, top, bottom)
        # F.pad expects padding in (left, right, top, bottom) format
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        # logger.debug(f"Padded input shape: {x.shape}") # Debug logging is now in forward
        return x


# Helper function to define the CGNet model based on config
def define_CGNet(opt):
    """
    Define the CGNet model (based on CascadedGaze architecture).

    Args:
        opt (dict): Configuration dictionary for the model.
            Expected structure:
            opt['arch']['type'] (str): 'CGNet'
            opt['arch']['args'] (dict): Dictionary of architecture-specific arguments.
                opt['arch']['args']['img_channel'] (int) # Expecting img_channel from config
                opt['arch']['args']['out_channels'] (int) # Expecting out_channels from config
                opt['arch']['args']['width'] (int) # Expecting base width from config
                opt['arch']['args']['middle_blk_num'] (int) # Expecting middle_blk_num from config
                opt['arch']['args']['enc_blk_nums'] (list) # Expecting encoder block counts per stage
                opt['arch']['args']['dec_blk_nums'] (list) # Expecting decoder block counts per stage
                opt['arch']['args']['GCE_CONVS_nums'] (list) # Expecting GCE conv counts per stage


    Returns:
        CGNet: The instantiated CGNet model.
    """
    # Access the architecture arguments from the 'arch' and 'args' keys
    # Corrected: Get the 'args' dictionary from the 'arch' dictionary
    opt_arch_args = opt.get('arch', {}).get('args', {}) # Get the 'args' dictionary, default to empty dict if 'arch' or 'args' is missing
    logger.info(f"Calling define_CGNet with architecture arguments: {opt_arch_args}") # Log the actual args being used

    # Instantiate the CGNet model using the arguments from the config
    model = CGNet(
        img_channel=opt_arch_args.get('img_channel', 3), # Get img_channel from args
        out_channels=opt_arch_args.get('out_channels', 3), # Get out_channels from args
        width=opt_arch_args.get('width', 60), # Get width from args
        middle_blk_num=opt_arch_args.get('middle_blk_num', 10), # Get middle_blk_num from args
        enc_blk_nums=opt_arch_args.get('enc_blk_nums', [2, 2, 2, 2]), # Get enc_blk_nums from args
        dec_blk_nums=opt_arch_args.get('dec_blk_nums', [2, 2, 2, 2]), # Get dec_blk_nums from args
        GCE_CONVS_nums=opt_arch_args.get('GCE_CONVS_nums', [3, 3, 2, 2]), # Get GCE_CONVS_nums from args
        # Note: use_gce, gce_stages, gce_reduction, upscale from previous CGNet are not parameters
        # in this CascadedGaze architecture. They are handled implicitly by the block types and GCE_CONVS_nums.
    )
    return model

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This block requires you to have arch_util.py and local_arch.py in the same directory
    # from arch_util import measure_inference_speed # Assuming measure_inference_speed is in arch_util.py
    # from ptflops import get_model_complexity_info # Uncomment if you have ptflops installed

    print("--- Testing CGNet (CascadedGaze architecture) instantiation ---")

    # Example parameters based on the provided __main__ block
    img_channel_test = 3
    width_test = 16 # Use a smaller width for testing
    enc_blks_test = [1, 1, 1, 1] # Reduced blocks for testing
    middle_blk_num_test = 2 # Reduced middle blocks for testing
    dec_blks_test = [1, 1, 1] # Adjusted decoder blocks for testing (should be num_encoder_stages - 1)
    GCE_CONVS_nums_test = [2, 2, 2, 2] # Keeping same for testing

    # Create a dummy config dictionary for testing define_CGNet
    dummy_config = {
        'arch': { # Nested structure matching the expected config
            'type': 'CGNet', # Should match the class name returned by define_CGNet
            'args': { # Nested 'args' dictionary
                'img_channel': img_channel_test,
                'out_channels': img_channel_test, # Assuming output channels match input for denoising
                'width': width_test,
                'middle_blk_num': middle_blk_num_test,
                'enc_blk_nums': enc_blks_test,
                dec_blk_nums: dec_blks_test, # Use adjusted dec_blk_nums
                'GCE_CONVS_nums': GCE_CONVS_nums_test,
            }
        }
    }

    try:
        # Define the network using the dummy config
        net = define_CGNet(dummy_config)
        print("\nSuccessfully defined CGNet model:")
        # print(net) # Uncomment to print the full model structure

        # Test with a dummy input tensor (e.g., 512x512 as per your current data)
        inp_shape_test = (img_channel_test, 512, 512)
        data = torch.randn((1, *inp_shape_test))
        print(f"\nTesting forward pass with input shape: {data.shape}")
        with torch.no_grad():
             out = net(data)
        print(f"Output shape: {out.shape}")
        # Check if output shape matches original input shape after cropping
        assert out.shape[-2:] == inp_shape_test[-2:], "Output spatial size does not match input spatial size after forward pass."
        print("Output spatial size matches input spatial size.")


        # # Uncomment if ptflops is installed
        # print("\n--- Model Complexity Info ---")
        # macs, params = get_model_complexity_info(net, inp_shape_test, verbose=False, print_per_layer_stat=False)
        # print(f"MACs: {macs}, Params: {params}")

        # # Uncomment if measure_inference_speed is available and you want to test speed
        # print("\n--- Inference Speed Measurement ---")
        # device = "cpu"
        # if torch.cuda.is_available():
        #     device = "cuda"
        # print(f"Using device: {device}")
        # data_device = torch.randn((1, *inp_shape_test)).to(device)
        # measure_inference_speed(net.to(device), (data_device,), max_iter=100, log_interval=20) # Reduced max_iter for quicker test


    except Exception as e:
        logger.error(f"\nAn error occurred during example usage: {e}")
        print("Please ensure all required classes (SimpleGate, etc.) and imports are correct.")

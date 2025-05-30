import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ObservationConvEncoder(nn.Module):
    """
    Convolutional encoder for processing observation inputs.
    Uses 1D convolutions to transform input features into a latent representation.
    
    Args:
        input_channels (int): Number of input feature channels
        output_channels (int): Number of output feature channels
        ie_num (int): Parameter controlling the encoder architecture (0-5)
    """
    def __init__(self, input_channels, output_channels, ie_num=2):
        super(ObservationConvEncoder, self).__init__()
        self.ie_num = ie_num
        # Define convolutional layers with consistent padding to maintain dimensions
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through the convolutional encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels]
            
        Returns:
            torch.Tensor: Encoded features of shape [batch_size, output_channels]
        """
        # Add channel dimension for 1D convolution
        x = x.unsqueeze(2)
        
        # Apply different network architectures based on the ie_num parameter
        # Options 0-2: Use ReLU activations
        # Options 3-5: No activation functions
        if self.ie_num == 0:
            x = F.relu(self.conv1(x))
        elif self.ie_num == 1:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        elif self.ie_num == 2:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        elif self.ie_num == 3:
            x = self.conv1(x)
        elif self.ie_num == 4:
            x = self.conv1(x)
            x = self.conv2(x)
        elif self.ie_num == 5:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        else:
            raise RuntimeError('Invalid ie_num!')

        # Remove the added channel dimension
        x = x.squeeze(2)
        return x
    
class ObservationEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for processing observation inputs.
    Transforms input features into a latent representation using fully connected layers.
    
    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        ie_num (int): Parameter controlling the encoder architecture (not used directly)
    """
    def __init__(self, input_dim, output_dim, ie_num):
        super(ObservationEncoder, self).__init__()
        ie_num = 4  # Fixed expansion factor
        
        # Create a 3-layer MLP with ReLU activations
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*ie_num),
            nn.ReLU(),
            nn.Linear(input_dim*ie_num, input_dim*ie_num),
            nn.ReLU(),
            nn.Linear(input_dim*ie_num, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the MLP encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Encoded features of shape [batch_size, output_dim]
        """
        return self.mlp(x)

class LatentSpaceInterpolator(nn.Module):
    """
    Interpolates between two latent space representations to generate intermediate frames.
    Supports various interpolation initialization strategies and usage modes.
    
    Args:
        dimension_num (int): Dimension of the latent space
        block_num (int): Number of intermediate frames to generate
        interpolation_init (int): Interpolation initialization strategy (0-12)
        interpolation_usage (int): Interpolation usage mode (0-1)
    """
    def __init__(self, dimension_num, block_num, interpolation_init, interpolation_usage):
        super(LatentSpaceInterpolator, self).__init__()
        self.block_num = block_num
        self.dimension_num = dimension_num
        # Linear layer to generate interpolation weights dynamically
        self.alpha_generator = nn.Linear(dimension_num, block_num)
        # Sigmoid ensures interpolation weights are between 0 and 1
        self.sigmoid = nn.Sigmoid()
        self.interpolation_init = interpolation_init
        self.interpolation_usage = interpolation_usage
        
    def forward(self, x1, x2):
        """
        Generate interpolated frames between two input representations.
        
        Args:
            x1 (torch.Tensor): First input tensor of shape [batch_size, dimension_num]
            x2 (torch.Tensor): Second input tensor of shape [batch_size, dimension_num]
            
        Returns:
            torch.Tensor: Interpolated frames of shape [batch_size, block_num, dimension_num]
        """
        batch_size, *_ = x1.shape
        x_combined = torch.stack([x1, x2], dim=0)
        
        # Generate interpolation weights based on selected initialization strategy
        if self.interpolation_init == 0:
            # Initialize with ones
            alphas = self.sigmoid(self.alpha_generator(
                torch.ones(batch_size, self.dimension_num).to(x1.device)))
        elif self.interpolation_init == 1:
            # Initialize with 0.5
            alphas = self.sigmoid(self.alpha_generator(
                torch.full((batch_size, self.dimension_num), 0.5).to(x1.device)))
        elif self.interpolation_init == 2:
            # Initialize with 2.0
            alphas = self.sigmoid(self.alpha_generator(
                torch.full((batch_size, self.dimension_num), 2).to(x1.device)))
        elif self.interpolation_init == 3:
            # Initialize with zeros
            alphas = self.sigmoid(self.alpha_generator(
                torch.zeros(batch_size, self.dimension_num).to(x1.device)))
        elif self.interpolation_init == 4:
            # Initialize with random values between 0 and 1
            alphas = self.sigmoid(self.alpha_generator(
                torch.rand((batch_size, self.dimension_num)).to(x1.device)))
        elif self.interpolation_init == 5:
            # Return repeated first input
            return x1.unsqueeze(1).repeat(1, self.block_num, 1)
        elif self.interpolation_init == 6:
            # Return repeated second input
            return x2.unsqueeze(1).repeat(1, self.block_num, 1)
        elif self.interpolation_init == 7:
            # Linear interpolation from 1 to 6
            alphas = torch.linspace(1, 6, self.block_num).repeat(batch_size, 1).to(x1.device)
        elif self.interpolation_init == 8:
            # Linear interpolation from 6 to 1
            alphas = torch.linspace(6, 1, self.block_num).repeat(batch_size, 1).to(x1.device)
        elif self.interpolation_init == 9:
            # Quadratic interpolation from 1 to 6
            alphas = torch.linspace(1, 6, self.block_num).repeat(batch_size, 1).to(x1.device)
            alphas = alphas ** 2
        elif self.interpolation_init == 10:
            # Quadratic interpolation from 6 to 1
            alphas = torch.linspace(6, 1, self.block_num).repeat(batch_size, 1).to(x1.device)
            alphas = alphas ** 2
        elif self.interpolation_init == 11:
            # V-shaped interpolation: 6→1→6
            half_block = self.block_num // 2
            alphas_first_half = torch.linspace(6, 1, half_block)
            alphas_second_half = torch.linspace(1, 6, self.block_num - half_block)
            alphas = torch.cat([alphas_first_half, alphas_second_half]).repeat(batch_size, 1).to(x1.device)
        elif self.interpolation_init == 12:
            # Split approach: first half x1, second half x2
            half_block = self.block_num // 2
            x1_half = x1.unsqueeze(1).repeat(1, half_block, 1)
            x2_half = x2.unsqueeze(1).repeat(1, self.block_num - half_block, 1)
            alphas = torch.cat([x1_half, x2_half], dim=1)
        else:
            raise RuntimeError('Invalid interpolation_init!')
        
        # Add dimension for broadcasting
        alphas = alphas.unsqueeze(-1)
        
        # Apply interpolation formula based on usage mode
        if self.interpolation_usage == 0 or self.interpolation_usage == 1:
            # Linear interpolation: (1-α)*x1 + α*x2
            interpolated_frames = (
                1 - alphas) * x_combined[0].unsqueeze(1) + alphas * x_combined[1].unsqueeze(1)
        else:
            raise RuntimeError('Invalid interpolation_usage!')

        return interpolated_frames

class TransformerBlock(nn.Module):
    """
    Wrapper for PyTorch's TransformerEncoder to process sequences of features.
    
    Args:
        dim (int): Dimension of the input features
        num_heads (int): Number of attention heads in the transformer
        num_layers (int): Number of transformer layers
    """
    def __init__(self, dim, num_heads, num_layers):
        super(TransformerBlock, self).__init__()
        encoder_layers = TransformerEncoderLayer(dim, num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, dim]
            
        Returns:
            torch.Tensor: Transformed output of the same shape
        """
        x = self.transformer(x)
        return x
    
class ActionPredictor(nn.Module):
    """
    Main model for action prediction by interpolating between observations and 
    optionally refining with transformer blocks.
    
    Args:
        args: Configuration object containing model parameters
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output features
        block_num (int): Number of intermediate frames to generate
        num_transformer_blocks (int): Number of transformer blocks to use
    """
    def __init__(self, args, input_dim, output_dim, block_num, num_transformer_blocks=1):
        super(ActionPredictor, self).__init__()
        
        self.module_kind = args.module_kind
        self.encoder_kind = args.encoder_kind

        # Initialize encoder based on specified type
        if self.encoder_kind == 'linear':
            self.encoder = ObservationEncoder(input_dim, output_dim, args.ie_num)
        else:
            self.encoder = ObservationConvEncoder(input_dim, output_dim, args.ie_num)
        
        # Initialize interpolator
        self.interpolator = LatentSpaceInterpolator(
            output_dim, block_num, args.interpolation_init, args.interpolation_usage)

        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            output_dim, num_heads=8, num_layers=args.transformer_num) for _ in range(num_transformer_blocks)])

    def forward(self, x1, x2):
        """
        Forward pass based on the selected module configuration.
        
        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Predicted intermediate frames
        """
        if self.module_kind == 'i':
            # Interpolation only
            interpolated_frames = self.interpolator(x1, x2)
            return interpolated_frames
            
        elif self.module_kind == 'e+i':
            # Encode then interpolate
            x1_encoded = self.encoder(x1)
            x2_encoded = self.encoder(x2)
            interpolated_frames = self.interpolator(x1_encoded, x2_encoded)
            return interpolated_frames
            
        elif self.module_kind == 'i+t':
            # Interpolate then transform
            interpolated_frames = self.interpolator(x1, x2)
            transformer_input = interpolated_frames
            
            # Apply transformer blocks sequentially
            for transformer_block in self.transformer_blocks:
                transformer_input = transformer_block(transformer_input)
                
            # Residual connection
            output = transformer_input + interpolated_frames
            return output
            
        else:  # Full pipeline: 'e+i+t'
            # Encode inputs
            x1_encoded = self.encoder(x1)
            x2_encoded = self.encoder(x2)
            
            # Interpolate between encoded representations
            interpolated_frames = self.interpolator(x1_encoded, x2_encoded)
            transformer_input = interpolated_frames
            
            # Apply transformer blocks sequentially
            for transformer_block in self.transformer_blocks:
                transformer_input = transformer_block(transformer_input)
                
            # Residual connection
            output = transformer_input + interpolated_frames
            return output

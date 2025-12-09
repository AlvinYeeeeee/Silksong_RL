"""
Neural network architectures for Deep Q-Learning.

This module provides CNN feature extractors and Q-network heads
for pixel-based reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# =============================================================================
# Weight Initialization
# =============================================================================

def initialize_weights(module, activation='relu'):
    """
    Initialize network weights using appropriate schemes.
    
    Args:
        module: PyTorch module to initialize
        activation: Activation function name for gain calculation
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, a=1e-2, mode="fan_out", nonlinearity=activation
        )
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, FactorizedNoisyLayer):
        module.init_parameters()
        module.sample_noise()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.constant_(module.bias, 0)


# =============================================================================
# Noisy Networks for Exploration
# =============================================================================

class FactorizedNoisyLayer(nn.Module):
    """
    Linear layer with factorized Gaussian noise for exploration.
    
    Implements the Noisy Networks approach where exploration is driven
    by parametric noise rather than epsilon-greedy.
    
    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018
    """
    
    def __init__(self, input_dim, output_dim, noise_std=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        
        # Mean parameters (learned)
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        
        # Noise scale parameters (learned)
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        
        # Factorized noise buffers
        self.register_buffer('noise_in', torch.empty(input_dim))
        self.register_buffer('noise_out', torch.empty(output_dim))
        
        self._exploration_enabled = True
        self.init_parameters()
        self.sample_noise()
    
    @staticmethod
    def _scale_noise(x):
        """Apply signed sqrt transformation to noise."""
        return x.normal_().sign().mul(x.abs().sqrt())
    
    def init_parameters(self):
        """Initialize parameters with uniform distribution."""
        bound = 1.0 / np.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.noise_std / np.sqrt(self.input_dim))
        self.bias_sigma.data.fill_(self.noise_std / np.sqrt(self.output_dim))
    
    def sample_noise(self):
        """Sample new factorized noise."""
        self.noise_in.copy_(self._scale_noise(self.noise_in))
        self.noise_out.copy_(self._scale_noise(self.noise_out))
    
    def set_exploration(self, enabled):
        """Enable or disable noise for exploration."""
        self._exploration_enabled = enabled
    
    def forward(self, x):
        if self._exploration_enabled:
            # Compute noisy weights via outer product
            weight = self.weight_mu + self.weight_sigma * self.noise_out.ger(self.noise_in)
            bias = self.bias_mu + self.bias_sigma * self.noise_out.clone()
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# =============================================================================
# Residual Building Block
# =============================================================================

class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    
    def __init__(self, in_ch, out_ch, stride=1, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        
        self.main_path = nn.Sequential(
            activation,
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        
        # Shortcut connection
        if stride > 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        return self.main_path(x) + self.skip(x)


# =============================================================================
# Feature Extractor Base Class
# =============================================================================

class VisualEncoder(nn.Module):
    """Base class for visual feature extractors."""
    
    def __init__(self, spatial_shape, input_channels, activation='relu', use_spectral_norm=False):
        super().__init__()
        self._activation_name = activation
        self.feature_dim = None  # To be set by subclasses
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")


# =============================================================================
# CNN Architectures
# =============================================================================

class ResNetEncoder(VisualEncoder):
    """
    ResNet-style feature extractor with residual connections.
    Suitable for complex visual environments.
    """
    
    def __init__(self, spatial_shape, input_channels, activation='relu', use_spectral_norm=False):
        super().__init__(spatial_shape, input_channels, activation, use_spectral_norm)
        
        act_fn = nn.LeakyReLU(inplace=True) if activation == 'leaky_relu' else nn.ReLU(inplace=True)
        
        # Calculate output spatial dimensions (32x downsampling)
        out_spatial = np.array(spatial_shape, dtype=int) // 32
        
        final_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(48, 48, activation=act_fn),
            nn.Conv2d(48, 96, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(96, 96, activation=act_fn),
            nn.Conv2d(96, 160, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(160, 160, activation=act_fn),
            ResidualBlock(160, 160, activation=act_fn),
            nn.Conv2d(160, 256, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, 256, activation=act_fn),
            final_conv,
            nn.Flatten(),
        )
        
        self.feature_dim = 256 * int(np.prod(out_spatial))
        
        # Initialize weights
        for module in self.modules():
            initialize_weights(module, activation)
    
    def forward(self, x):
        return self.encoder(x)


class ConvEncoder(VisualEncoder):
    """
    Simple 5-layer CNN feature extractor.
    Efficient for most game environments.
    """
    
    def __init__(self, spatial_shape, input_channels, activation='relu', use_spectral_norm=False):
        super().__init__(spatial_shape, input_channels, activation, use_spectral_norm)
        
        act_fn = nn.LeakyReLU(inplace=True) if activation == 'leaky_relu' else nn.ReLU(inplace=True)
        
        # 5 stride-2 convolutions = 32x downsampling
        out_spatial = np.array(spatial_shape, dtype=int) // 32
        
        final_conv = nn.Conv2d(160, 320, kernel_size=3, stride=2, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            act_fn,
            final_conv,
            act_fn,
            nn.Flatten(),
        )
        
        self.feature_dim = 320 * int(np.prod(out_spatial))
        
        for module in self.modules():
            initialize_weights(module, activation)
    
    def forward(self, x):
        return self.encoder(x)


class CompactEncoder(VisualEncoder):
    """
    Lightweight CNN with larger strides for faster processing.
    Trade-off: less spatial detail for speed.
    """
    
    def __init__(self, spatial_shape, input_channels, activation='relu', use_spectral_norm=False):
        super().__init__(spatial_shape, input_channels, activation, use_spectral_norm)
        
        act_fn = nn.LeakyReLU(inplace=True) if activation == 'leaky_relu' else nn.ReLU(inplace=True)
        
        out_spatial = np.array(spatial_shape, dtype=int) // 32
        
        final_conv = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=4),
            act_fn,
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            act_fn,
            final_conv,
            act_fn,
            nn.Flatten()
        )
        
        self.feature_dim = 128 * int(np.prod(out_spatial))
        
        for module in self.modules():
            initialize_weights(module, activation)
    
    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# Q-Network Heads
# =============================================================================

class QNetworkBase(nn.Module):
    """Base class for Q-network heads."""
    
    def __init__(self, encoder, num_actions, activation='relu', noisy=False, use_spectral_norm=False):
        super().__init__()
        
        if noisy and use_spectral_norm:
            raise ValueError("Spectral norm is incompatible with noisy networks")
        
        self._noisy_layers = nn.ModuleList()
        self._fc_layers = nn.ModuleList()
        self._linear_class = FactorizedNoisyLayer if noisy else nn.Linear
        self.encoder = encoder
        
        if activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def resample_noise(self):
        """Resample noise for all noisy layers."""
        for layer in self._noisy_layers:
            layer.sample_noise()
    
    def set_exploration(self, enabled):
        """Enable or disable exploration noise."""
        for layer in self._noisy_layers:
            layer.set_exploration(enabled)
    
    def reinitialize_fc(self):
        """Reinitialize fully-connected layer parameters."""
        count = 0
        for layer in self._fc_layers:
            initialize_weights(layer, self.encoder._activation_name)
            count += 1
        print(f'Reinitialized {count} fully-connected layers')


class StandardQNetwork(QNetworkBase):
    """Single-path Q-network with one hidden layer."""
    
    def __init__(self, encoder, num_actions, activation='relu', noisy=False, use_spectral_norm=False):
        super().__init__(encoder, num_actions, activation, noisy, use_spectral_norm)
        
        self.hidden = self._linear_class(encoder.feature_dim, 800)
        self.output = self._linear_class(800, num_actions)
        
        if use_spectral_norm:
            self.hidden = spectral_norm(self.hidden)
        
        if noisy:
            self._noisy_layers.extend([self.hidden, self.output])
        
        self._fc_layers = nn.ModuleList([self.hidden, self.output])
        self.reinitialize_fc()
    
    def forward(self, x, **kwargs):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        hidden = self.activation(self.hidden(features))
        return self.output(hidden)


class DuelingQNetwork(QNetworkBase):
    """
    Dueling DQN architecture separating value and advantage streams.
    
    Q(s,a) = V(s) + A(s,a) - mean(A)
    
    Reference: Wang et al., "Dueling Network Architectures", ICML 2016
    """
    
    def __init__(self, encoder, num_actions, activation='relu', noisy=False, use_spectral_norm=False):
        super().__init__(encoder, num_actions, activation, noisy, use_spectral_norm)
        
        # Value stream: estimates V(s)
        self.value_hidden = self._linear_class(encoder.feature_dim, 512)
        self.value_out = self._linear_class(512, 1)
        
        # Advantage stream: estimates A(s,a)
        self.advantage_hidden = self._linear_class(encoder.feature_dim, 512)
        self.advantage_out = self._linear_class(512, num_actions)
        
        if use_spectral_norm:
            self.value_hidden = spectral_norm(self.value_hidden)
            self.advantage_hidden = spectral_norm(self.advantage_hidden)
        
        if noisy:
            self._noisy_layers.extend([
                self.value_hidden, self.value_out,
                self.advantage_hidden, self.advantage_out
            ])
        
        self._fc_layers = nn.ModuleList([
            self.value_hidden, self.advantage_hidden,
            self.value_out, self.advantage_out
        ])
        
        self.reinitialize_fc()
    
    def forward(self, x, advantage_only=False, **kwargs):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        
        # Compute advantage
        adv_hidden = self.activation(self.advantage_hidden(features))
        advantage = self.advantage_out(adv_hidden)
        
        if advantage_only:
            return advantage
        
        # Compute value
        val_hidden = self.activation(self.value_hidden(features))
        value = self.value_out(val_hidden)
        
        # Combine: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

NoisyLinear = FactorizedNoisyLayer
BasicBlock = ResidualBlock
AbstractExtractor = VisualEncoder
ResidualExtractor = ResNetEncoder
SimpleExtractor = ConvEncoder
TinyExtractor = CompactEncoder
AbstractFullyConnected = QNetworkBase
SinglePathMLP = StandardQNetwork
DuelingMLP = DuelingQNetwork
param_init = initialize_weights

# Method aliases for backward compatibility
DuelingQNetwork.reset_noise = DuelingQNetwork.resample_noise
DuelingQNetwork.noise_mode = DuelingQNetwork.set_exploration
DuelingQNetwork.reset_params = DuelingQNetwork.reinitialize_fc
StandardQNetwork.reset_noise = StandardQNetwork.resample_noise
StandardQNetwork.noise_mode = StandardQNetwork.set_exploration
StandardQNetwork.reset_params = StandardQNetwork.reinitialize_fc
FactorizedNoisyLayer.reset_param = FactorizedNoisyLayer.init_parameters
FactorizedNoisyLayer.reset_noise = FactorizedNoisyLayer.sample_noise
FactorizedNoisyLayer.noise_mode = FactorizedNoisyLayer.set_exploration
VisualEncoder.units = property(lambda self: self.feature_dim)
VisualEncoder.activation_name = property(lambda self: self._activation_name)
ConvEncoder.units = property(lambda self: self.feature_dim)
ConvEncoder.activation_name = property(lambda self: self._activation_name)
ResNetEncoder.units = property(lambda self: self.feature_dim)
ResNetEncoder.activation_name = property(lambda self: self._activation_name)
CompactEncoder.units = property(lambda self: self.feature_dim)
CompactEncoder.activation_name = property(lambda self: self._activation_name)


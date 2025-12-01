"""
Latent Diffusion Model for Text-to-Image Generation

This module implements a UNet-based diffusion model that operates in the latent space
of images (256 channels, 7x7 spatial resolution) rather than pixel space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LatentDiffusionUNet(nn.Module):
    """
    Latent Diffusion Model with UNet architecture

    Operates on image latent representations (B, 256, 7, 7) conditioned on text embeddings.
    This is much more efficient than pixel-space diffusion and preserves spatial structure.
    """

    def __init__(self, latent_channels=256, latent_size=7, condition_dim=512,
                 num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (will be moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Precompute values for diffusion
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # UNet noise predictor
        self.unet = ConditionalUNet2D(
            in_channels=latent_channels,
            out_channels=latent_channels,
            condition_dim=condition_dim,
            base_channels=128,
            num_timesteps=num_timesteps
        )

    def forward_diffusion(self, x_0, t):
        """
        Add noise to latent representations (forward diffusion process)

        Args:
            x_0: (B, C, H, W) - clean latent representations
            t: (B,) - timestep indices [0, num_timesteps)

        Returns:
            x_t: (B, C, H, W) - noisy latents
            noise: (B, C, H, W) - the noise that was added
        """
        # Get noise schedule coefficients
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Add noise according to schedule: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def predict_noise(self, x_t, t, condition):
        """
        Predict noise from noisy latents using UNet

        Args:
            x_t: (B, C, H, W) - noisy latent
            t: (B,) - timestep indices
            condition: (B, condition_dim) - text embedding for conditioning

        Returns:
            noise_pred: (B, C, H, W) - predicted noise
        """
        return self.unet(x_t, t, condition)

    @torch.no_grad()
    def sample(self, batch_size, device, condition, num_inference_steps=50):
        """
        Generate latent representations via reverse diffusion (DDIM sampling)

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            condition: (B, condition_dim) - text embeddings for conditioning
            num_inference_steps: Number of denoising steps (fewer = faster, more = better quality)

        Returns:
            x_0: (B, C, H, W) - generated clean latents
        """
        # Start from random noise
        x_t = torch.randn(batch_size, self.latent_channels,
                         self.latent_size, self.latent_size, device=device)

        # DDIM sampling: use subset of timesteps for faster sampling
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(reversed(range(0, self.num_timesteps, step_size)))

        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            # Predict noise with text conditioning
            noise_pred = self.predict_noise(x_t, t_batch, condition)

            # DDIM update
            alpha_t = self.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # Predict x_0 from x_t and noise
            pred_x_0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_x_t = torch.sqrt(1 - alpha_t_prev) * noise_pred

            # DDIM update: x_{t-1} = sqrt(alpha_{t-1}) * pred_x_0 + direction
            x_t = torch.sqrt(alpha_t_prev) * pred_x_0 + dir_x_t

        return x_t


class ConditionalUNet2D(nn.Module):
    """
    UNet architecture for noise prediction with text conditioning

    Architecture:
    - Encoder: Downsample with ResBlocks
    - Middle: ResBlocks + Attention
    - Decoder: Upsample with ResBlocks + Skip connections
    """

    def __init__(self, in_channels=256, out_channels=256, condition_dim=512,
                 base_channels=128, num_timesteps=1000):
        super().__init__()

        self.num_timesteps = num_timesteps

        # Timestep embedding
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Condition projection (text embedding → same dim as time embedding)
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (no downsampling since we're already at 7x7)
        self.encoder = nn.ModuleList([
            ResBlockWithCondition(base_channels, base_channels * 2, time_embed_dim),
            ResBlockWithCondition(base_channels * 2, base_channels * 4, time_embed_dim),
        ])

        # Middle (bottleneck with attention)
        mid_channels = base_channels * 4
        self.middle = nn.ModuleList([
            ResBlockWithCondition(mid_channels, mid_channels, time_embed_dim),
            AttentionBlock(mid_channels),
            ResBlockWithCondition(mid_channels, mid_channels, time_embed_dim)
        ])

        # Decoder (with skip connections)
        self.decoder = nn.ModuleList([
            ResBlockWithCondition(mid_channels * 2, base_channels * 2, time_embed_dim),  # *2 for skip
            ResBlockWithCondition(base_channels * 2 * 2, base_channels, time_embed_dim),  # *2 for skip
        ])

        # Output projection
        self.out_norm = nn.GroupNorm(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, t, condition):
        """
        Args:
            x: (B, C, H, W) - noisy latent input
            t: (B,) - timestep indices
            condition: (B, condition_dim) - text embeddings

        Returns:
            noise_pred: (B, C, H, W) - predicted noise
        """
        # Embed timestep and condition
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        c_emb = self.condition_proj(condition)  # (B, time_embed_dim)

        # Combine time and condition embeddings
        cond = t_emb + c_emb  # (B, time_embed_dim)

        # Initial conv
        h = self.init_conv(x)  # (B, base_channels, H, W)

        # Encoder with skip connections
        skips = []
        for block in self.encoder:
            h = block(h, cond)
            skips.append(h)

        # Middle
        h = self.middle[0](h, cond)
        h = self.middle[1](h)
        h = self.middle[2](h, cond)

        # Decoder with skip connections
        for block in self.decoder:
            h = torch.cat([h, skips.pop()], dim=1)  # Concatenate skip connection
            h = block(h, cond)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (B,) - timestep indices

        Returns:
            embeddings: (B, dim) - sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResBlockWithCondition(nn.Module):
    """Residual block with condition injection via FiLM (Feature-wise Linear Modulation)"""

    def __init__(self, in_channels, out_channels, condition_dim, groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Condition projection (FiLM: scale and shift)
        self.condition_proj = nn.Linear(condition_dim, out_channels * 2)

        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, condition):
        """
        Args:
            x: (B, in_channels, H, W)
            condition: (B, condition_dim) - combined time + text embedding

        Returns:
            out: (B, out_channels, H, W)
        """
        residual = self.skip(x)

        # First conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Inject condition via FiLM
        cond = self.condition_proj(condition)  # (B, out_channels * 2)
        scale, shift = cond.chunk(2, dim=1)  # (B, out_channels) each
        scale = scale[:, :, None, None]  # (B, out_channels, 1, 1)
        shift = shift[:, :, None, None]  # (B, out_channels, 1, 1)

        h = h * (1 + scale) + shift  # FiLM modulation

        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global dependencies"""

    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        h = self.norm(x)

        # QKV projection
        qkv = self.qkv(h)  # (B, C*3, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)  # (B, 3, num_heads, head_dim, H*W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, H*W, head_dim)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h = torch.matmul(attn, v)  # (B, num_heads, H*W, head_dim)
        h = h.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, H*W)
        h = h.reshape(B, C, H, W)  # (B, C, H, W)

        # Output projection
        h = self.proj(h)

        return h + residual

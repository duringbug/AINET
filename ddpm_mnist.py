import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic convolutional block with GroupNorm"""

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = F.silu(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        # Residual connection
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """
    Simple UNet for DDPM
    Architecture optimized for 28x28 MNIST images
    28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=64,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        # Level 1: 28x28, 64 channels
        self.down1 = Block(base_channels, base_channels, time_emb_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        # Level 2: 14x14, 128 channels
        self.down2 = Block(base_channels, base_channels * 2, time_emb_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck: 7x7, 128 channels
        self.bottleneck = Block(base_channels * 2, base_channels * 2, time_emb_dim)

        # Decoder
        # Level 2: 7x7 -> 14x14, 128 -> 64 channels
        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = Block(base_channels * 4, base_channels, time_emb_dim)  # *4 because of skip connection

        # Level 1: 14x14 -> 28x28, 64 -> 64 channels
        self.upsample1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = Block(base_channels * 2, base_channels, time_emb_dim)  # *2 because of skip connection

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial convolution
        x = self.init_conv(x)  # [B, 64, 28, 28]

        # Encoder
        skip1 = self.down1(x, t)  # [B, 64, 28, 28]
        x = self.downsample1(skip1)  # [B, 64, 14, 14]

        skip2 = self.down2(x, t)  # [B, 128, 14, 14]
        x = self.downsample2(skip2)  # [B, 128, 7, 7]

        # Bottleneck
        x = self.bottleneck(x, t)  # [B, 128, 7, 7]

        # Decoder
        x = self.upsample2(x)  # [B, 128, 14, 14]
        x = torch.cat([x, skip2], dim=1)  # [B, 256, 14, 14]
        x = self.up2(x, t)  # [B, 64, 14, 14]

        x = self.upsample1(x)  # [B, 64, 28, 28]
        x = torch.cat([x, skip1], dim=1)  # [B, 128, 28, 28]
        x = self.up1(x, t)  # [B, 64, 28, 28]

        # Final convolution
        x = self.final_conv(x)  # [B, 1, 28, 28]

        return x


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model

    Based on the paper:
    "Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
    """

    def __init__(
        self,
        model,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device='cuda'
    ):
        super().__init__()

        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        # Create beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

        # Pre-calculate useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to the input image
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior(self, x_start, x_t, t):
        """Calculate the posterior mean and variance"""
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start +
            self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t][:, None, None, None]

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t, t):
        """
        Calculate the mean and variance for the reverse process
        p(x_{t-1} | x_t)
        """
        # Predict noise using the model
        predicted_noise = self.model(x_t, t)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Clip x_start to [-1, 1]
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Calculate posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start, x_t, t
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Sample from the reverse process: p(x_{t-1} | x_t)
        """
        batch_size = x_t.shape[0]

        # Get model predictions
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t)

        # No noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(batch_size, 1, 1, 1)

        # Sample from the distribution
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size, channels=1, image_size=28):
        """
        Generate samples by running the reverse diffusion process
        """
        self.model.eval()

        # Start from random noise
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        # Run reverse diffusion
        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            x = self.p_sample(x, t_batch)

        return x

    def forward(self, x_start):
        """
        Training forward pass
        Calculate the loss for training
        """
        batch_size = x_start.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(self.device)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Add noise to images (forward diffusion)
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.model(x_noisy, t)

        # Calculate MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss


def train_ddpm(
    ddpm,
    train_loader,
    num_epochs=50,
    learning_rate=2e-4,
    save_dir='outputs/ddpm_mnist',
    sample_freq=5,
    device='cuda'
):
    """Train the DDPM model"""

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)

    # Optimizer
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=learning_rate)

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(num_epochs):
        ddpm.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, _ in pbar:
            images = images.to(device)

            # Normalize images to [-1, 1]
            images = (images - 0.5) * 2.0

            # Forward pass
            loss = ddpm(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}')

        # Sample and save images
        if (epoch + 1) % sample_freq == 0:
            logger.info(f'Generating samples at epoch {epoch+1}...')

            # Generate samples
            samples = ddpm.sample(batch_size=64, channels=1, image_size=28)

            # Denormalize from [-1, 1] to [0, 1]
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            # Save grid of images
            grid = make_grid(samples, nrow=8, padding=2)
            save_image(grid, save_dir / 'samples' / f'sample_epoch_{epoch+1}.png')

            logger.info(f'Samples saved to {save_dir / "samples" / f"sample_epoch_{epoch+1}.png"}')

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / 'checkpoints' / f'ddpm_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

    # Save final model
    final_path = save_dir / 'ddpm_final.pt'
    torch.save({
        'model_state_dict': ddpm.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f'Final model saved to {final_path}')

    return ddpm


def main():
    """Main training function"""

    # Configuration
    config = {
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 2e-4,
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'image_size': 28,
        'save_dir': 'outputs/ddpm_mnist',
        'sample_freq': 5,  # Sample every N epochs
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load MNIST dataset
    logger.info('Loading MNIST dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    logger.info(f'Dataset size: {len(train_dataset)}')
    logger.info(f'Number of batches: {len(train_loader)}')

    # Create model
    logger.info('Creating DDPM model...')
    unet = UNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=64,
    ).to(device)

    ddpm = DDPM(
        model=unet,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )

    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {total_params:,} ({trainable_params:,} trainable)')

    # Train model
    logger.info('Starting training...')
    train_ddpm(
        ddpm=ddpm,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        sample_freq=config['sample_freq'],
        device=device
    )

    logger.info('Training completed!')

    # Generate final samples
    logger.info('Generating final samples...')
    ddpm.model.eval()
    samples = ddpm.sample(batch_size=64, channels=1, image_size=28)
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)

    grid = make_grid(samples, nrow=8, padding=2)
    save_path = Path(config['save_dir']) / 'final_samples.png'
    save_image(grid, save_path)
    logger.info(f'Final samples saved to {save_path}')


if __name__ == '__main__':
    main()

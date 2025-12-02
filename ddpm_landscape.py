import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from PIL import Image
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LandscapeDataset(Dataset):
    """Dataset for loading landscape images from a folder"""

    def __init__(self, data_dir, image_size=64, transform=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_paths.extend(glob.glob(str(self.data_dir / '**' / ext), recursive=True))

        logger.info(f'Found {len(self.image_paths)} images in {data_dir}')

        if len(self.image_paths) == 0:
            raise ValueError(f'No images found in {data_dir}')

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, 0  # Return dummy label
        except Exception as e:
            logger.warning(f'Error loading image {img_path}: {e}')
            # Return a random image if loading fails
            return torch.randn(3, self.image_size, self.image_size), 0


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


class AttentionBlock(nn.Module):
    """Self-attention block"""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhci,bhcj->bhij', q, k) * scale, dim=-1)
        h = torch.einsum('bhij,bhcj->bhci', attn, v)

        # Reshape back
        h = h.reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class UNet(nn.Module):
    """
    UNet for DDPM on landscape images
    Flexible architecture that adapts to different resolutions
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        use_attention=(False, True, True, False),
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.num_levels = len(channel_mult)

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
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attentions = nn.ModuleList()

        ch_in = base_channels
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult

            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(Block(ch_in, ch_out, time_emb_dim))
                ch_in = ch_out

            self.down_blocks.append(blocks)

            # Attention
            if use_attention[i]:
                self.down_attentions.append(AttentionBlock(ch_out))
            else:
                self.down_attentions.append(nn.Identity())

            # Downsample (except for the last level)
            if i < len(channel_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(ch_out, ch_out, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.down_samples.append(nn.Identity())

        # Bottleneck
        bottleneck_channels = base_channels * channel_mult[-1]
        self.bottleneck = nn.Sequential(
            Block(bottleneck_channels, bottleneck_channels, time_emb_dim),
            AttentionBlock(bottleneck_channels),
            Block(bottleneck_channels, bottleneck_channels, time_emb_dim),
        )

        # Decoder
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            ch_out = base_channels * mult
            ch_in_skip = base_channels * channel_mult[-(i + 1)]

            # Upsample (except for the first level in decoder, which is the last in encoder)
            if i > 0:
                ch_in_current = base_channels * channel_mult[-(i)]
                self.up_samples.append(
                    nn.ConvTranspose2d(ch_in_current, ch_in_current, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())

            # Residual blocks
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                if j == 0:
                    # First block combines upsampled features with skip connection
                    if i > 0:
                        ch_in_current = base_channels * channel_mult[-(i)]
                    else:
                        ch_in_current = base_channels * channel_mult[-1]
                    blocks.append(Block(ch_in_current + ch_in_skip, ch_out, time_emb_dim))
                else:
                    blocks.append(Block(ch_out, ch_out, time_emb_dim))

            self.up_blocks.append(blocks)

            # Attention
            level_idx = len(channel_mult) - 1 - i
            if use_attention[level_idx]:
                self.up_attentions.append(AttentionBlock(ch_out))
            else:
                self.up_attentions.append(nn.Identity())

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
        x = self.init_conv(x)

        # Encoder
        skips = []
        for blocks, attn, downsample in zip(self.down_blocks, self.down_attentions, self.down_samples):
            for block in blocks:
                x = block(x, t)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck[0](x, t)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t)

        # Decoder
        for blocks, attn, upsample, skip in zip(
            self.up_blocks, self.up_attentions, self.up_samples, reversed(skips)
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)

            for block in blocks:
                x = block(x, t)

            x = attn(x)

        # Final convolution
        x = self.final_conv(x)

        return x


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

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

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

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
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, x_t, t):
        predicted_noise = self.model(x_t, t)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        x_start = torch.clamp(x_start, -1.0, 1.0)

        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start +
            self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        posterior_log_variance = self.posterior_log_variance_clipped[t][:, None, None, None]

        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t):
        batch_size = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(batch_size, 1, 1, 1)

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size, channels=3, image_size=64):
        self.model.eval()

        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            x = self.p_sample(x, t_batch)

        return x

    @torch.no_grad()
    def sample_ddim(self, batch_size, channels=3, image_size=64, ddim_steps=50, ddim_eta=0.0):
        """Generate images using DDIM sampling

        Args:
            batch_size: number of images to generate
            channels: number of channels (3 for RGB)
            image_size: size of generated images
            ddim_steps: number of steps for DDIM (fewer steps = faster)
            ddim_eta: stochasticity parameter (0=deterministic, 1=DDPM-like)

        Returns:
            Generated images tensor
        """
        self.model.eval()

        # Start from noise
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        # DDIM sampling: use subset of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Get alpha values
            alpha_cumprod_t = self.alphas_cumprod[t]

            # Predict x0 from xt and noise
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0).to(self.device)

            # Compute direction pointing to xt
            sigma_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise

            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir

            # Add noise (controlled by eta)
            if ddim_eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise

        return x

    def forward(self, x_start):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(self.device)
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss


def train_ddpm(
    ddpm,
    train_loader,
    num_epochs=200,
    learning_rate=2e-4,
    save_dir='outputs/ddpm_landscape',
    sample_freq=10,
    device='cuda',
    image_size=64
):
    """Train the DDPM model"""

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)

    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(num_epochs):
        ddpm.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, _ in pbar:
            images = images.to(device)
            images = (images - 0.5) * 2.0  # Normalize to [-1, 1]

            loss = ddpm(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}')

        scheduler.step()

        # Sample and save images
        if (epoch + 1) % sample_freq == 0:
            logger.info(f'Generating samples at epoch {epoch+1}...')
            samples = ddpm.sample(batch_size=16, channels=3, image_size=image_size)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            grid = make_grid(samples, nrow=4, padding=2)
            save_image(grid, save_dir / 'samples' / f'sample_epoch_{epoch+1}.png')

        # Save checkpoint
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_dir / 'checkpoints' / f'ddpm_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

    final_path = save_dir / 'ddpm_final.pt'
    torch.save({'model_state_dict': ddpm.model.state_dict()}, final_path)
    logger.info(f'Final model saved to {final_path}')

    return ddpm


def main():
    """Main training function"""

    config = {
        'data_dir': 'data/landscape',  # 修改为你的风景图片文件夹
        'batch_size': 32,  # 减小batch size因为图像更大
        'num_epochs': 200,
        'learning_rate': 2e-4,
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'image_size': 64,  # 可以改为128获得更高质量，但训练更慢
        'save_dir': 'outputs/ddpm_landscape',
        'sample_freq': 10,
        'base_channels': 128,
        'channel_mult': (1, 2, 2, 4),
        'num_res_blocks': 2,
        'use_attention': (False, True, True, False),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load dataset
    logger.info(f'Loading landscape images from {config["data_dir"]}...')

    try:
        train_dataset = LandscapeDataset(
            config['data_dir'],
            image_size=config['image_size']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True
        )

        logger.info(f'Dataset size: {len(train_dataset)}')
        logger.info(f'Number of batches: {len(train_loader)}')

    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        logger.info('Please make sure you have images in the data/landscape directory')
        logger.info('You can download a landscape dataset from Kaggle or use your own images')
        return

    # Create model
    logger.info('Creating DDPM model...')
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=config['base_channels'],
        channel_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
    ).to(device)

    ddpm = DDPM(
        model=unet,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )

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
        device=device,
        image_size=config['image_size']
    )

    logger.info('Training completed!')


def test_ddim_sampling(
    checkpoint_path='outputs/ddpm_landscape/ddpm_final.pt',
    save_dir='outputs/ddpm_landscape/ddim_samples',
    ddim_steps=50,
    num_samples=16,
    image_size=64,
    device='cuda'
):
    """Test DDIM sampling with a trained model

    Args:
        checkpoint_path: path to trained model checkpoint
        save_dir: directory to save generated samples
        ddim_steps: number of DDIM steps (e.g., 50, 100, 200)
        num_samples: number of images to generate
        image_size: size of generated images
        device: device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load model
    logger.info(f'Loading model from {checkpoint_path}...')
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        use_attention=(False, True, True, False),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])

    ddpm = DDPM(
        model=unet,
        num_timesteps=1000,
        device=device
    )

    logger.info(f'Model loaded successfully!')

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Test DDIM sampling
    import time

    logger.info(f'\nGenerating {num_samples} samples with DDIM-{ddim_steps}...')
    start_time = time.time()

    samples = ddpm.sample_ddim(
        batch_size=num_samples,
        channels=3,
        image_size=image_size,
        ddim_steps=ddim_steps,
        ddim_eta=0.0  # Deterministic sampling
    )

    elapsed_time = time.time() - start_time
    logger.info(f'Sampling completed in {elapsed_time:.2f}s ({elapsed_time/num_samples:.3f}s per image)')

    # Normalize and save
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)

    grid = make_grid(samples, nrow=4, padding=2)
    save_image(grid, save_dir / f'ddim_{ddim_steps}steps.png')
    logger.info(f'Saved to {save_dir / f"ddim_{ddim_steps}steps.png"}')

    # Test with different eta values (stochasticity)
    logger.info(f'\nTesting different eta values (stochasticity)...')
    for eta in [0.0, 0.5, 1.0]:
        logger.info(f'Generating samples with DDIM-50 and eta={eta}...')

        samples = ddpm.sample_ddim(
            batch_size=num_samples,
            channels=3,
            image_size=image_size,
            ddim_steps=50,
            ddim_eta=eta
        )

        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)

        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, save_dir / f'ddim_50steps_eta{eta}.png')
        logger.info(f'Saved to {save_dir / f"ddim_50steps_eta{eta}.png"}')

    logger.info(f'\nAll samples saved to {save_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DDPM for Landscape Images')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test_ddim'],
                       help='Mode: train or test_ddim')
    parser.add_argument('--checkpoint', type=str, default='outputs/ddpm_landscape/ddpm_final.pt',
                       help='Checkpoint path for testing')
    parser.add_argument('--ddim-steps', type=int, default=50,
                       help='Number of DDIM steps')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test_ddim':
        test_ddim_sampling(
            checkpoint_path=args.checkpoint,
            ddim_steps=args.ddim_steps,
            num_samples=args.num_samples,
            device=args.device
        )

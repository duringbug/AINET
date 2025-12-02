"""
MNIST 数字生成测试 - 简单的文本(数字)到图像生成
从数字标签（0-9）生成对应的手写数字图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
from pathlib import Path

# Import from step2.py
from ddpm_landscape import SinusoidalPositionEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleConditionedUNet(nn.Module):
    """简化的条件 UNet for MNIST (28x28, 1 channel)"""

    def __init__(self, label_emb_dim=64, time_emb_dim=128, base_channels=64):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Label embedding (0-9)
        self.label_emb = nn.Embedding(10, label_emb_dim)

        # Combine time + label
        self.cond_proj = nn.Linear(time_emb_dim + label_emb_dim, base_channels)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # 14x14
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # 7x7
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),  # 14x14
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, stride=2, padding=1),  # 28x28
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x, timestep, labels):
        """
        x: (B, 1, 28, 28)
        timestep: (B,)
        labels: (B,) - digit labels 0-9
        """
        # Get embeddings
        t_emb = self.time_mlp(timestep)  # (B, time_emb_dim)
        l_emb = self.label_emb(labels)   # (B, label_emb_dim)

        # Combine condition
        cond = torch.cat([t_emb, l_emb], dim=1)  # (B, time_emb_dim + label_emb_dim)
        cond = self.cond_proj(cond)[:, :, None, None]  # (B, base_channels, 1, 1)

        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + cond  # Add condition
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection

        out = self.dec1(d2)

        return out


class SimpleDDPM(nn.Module):
    """简化的 DDPM with DDIM support"""

    def __init__(self, model, num_timesteps=1000, device='cuda'):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta schedule
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Add noise"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def sample(self, labels, image_size=28, ddim_steps=50, ddim_eta=0.0):
        """Generate images from labels using DDIM sampling

        Args:
            labels: digit labels to generate
            image_size: output image size
            ddim_steps: number of steps for DDIM (default: 50)
            ddim_eta: DDIM stochasticity parameter (0=deterministic, 1=DDPM-like)
        """
        self.model.eval()
        batch_size = labels.shape[0]

        # Start from noise
        x = torch.randn(batch_size, 1, image_size, image_size).to(self.device)

        # DDIM sampling: use subset of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)

            # Predict noise
            predicted_noise = self.model(x, t_batch, labels)

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
            pred_dir = torch.sqrt(1 - alpha_cumprod_t_prev - ddim_eta**2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * predicted_noise

            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir

            # Add noise (controlled by eta)
            if ddim_eta > 0 and i < len(timesteps) - 1:
                sigma = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                noise = torch.randn_like(x)
                x = x + sigma * noise

        return x

    def forward(self, x_start, labels):
        """Training forward"""
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(self.device)
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, labels)

        loss = F.mse_loss(predicted_noise, noise)
        return loss


def train_mnist_ddpm(num_epochs=20, batch_size=128, device='cuda'):
    """训练 MNIST 条件扩散模型"""

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model
    unet = SimpleConditionedUNet().to(device)
    ddpm = SimpleDDPM(unet, num_timesteps=1000, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4)

    # Output
    output_dir = Path('outputs/mnist_ddpm')
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'samples').mkdir(exist_ok=True)

    logger.info(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Training
    for epoch in range(num_epochs):
        ddpm.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            loss = ddpm(images, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1} - Loss: {avg_loss:.4f}')

        # Generate samples
        if (epoch + 1) % 5 == 0:
            # Generate one sample for each digit 0-9 (multiple times)
            labels = torch.tensor([i for i in range(10) for _ in range(8)]).to(device)
            samples = ddpm.sample(labels)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            grid = make_grid(samples, nrow=8, padding=2)
            save_image(grid, output_dir / 'samples' / f'epoch_{epoch+1}.png')

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, output_dir / 'checkpoints' / f'checkpoint_{epoch+1}.pt')

    # Final
    torch.save(unet.state_dict(), output_dir / 'final_model.pt')
    logger.info("Training completed!")


def test_mnist_generation(checkpoint_path='outputs/mnist_ddpm/final_model.pt', device='cuda',
                         ddim_steps=50):
    """测试：输入数字，生成图像 (使用DDIM采样器)

    Args:
        checkpoint_path: path to model checkpoint
        device: device to use
        ddim_steps: number of steps for DDIM (default: 50)
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    unet = SimpleConditionedUNet().to(device)
    unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ddpm = SimpleDDPM(unet, num_timesteps=1000, device=device)

    logger.info(f"Model loaded! Using DDIM-{ddim_steps}steps sampler")

    # Generate all digits (0-9), multiple samples each
    output_dir = Path('outputs/mnist_test')
    output_dir.mkdir(exist_ok=True, parents=True)

    import time
    start_time = time.time()

    # Generate 10 samples for each digit
    all_samples = []
    for digit in range(10):
        labels = torch.tensor([digit] * 10).to(device)
        samples = ddpm.sample(labels, ddim_steps=ddim_steps)
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)
    grid = make_grid(all_samples, nrow=10, padding=2)
    save_image(grid, output_dir / f'all_digits_ddim{ddim_steps}.png')

    elapsed_time = time.time() - start_time
    logger.info(f"Generated images saved to {output_dir}")
    logger.info(f"Sampling time: {elapsed_time:.2f}s ({elapsed_time/100:.3f}s per image)")
    logger.info("Each row contains 10 samples of the same digit (0-9)")

    # Generate specific digits
    logger.info("\nGenerating specific digits...")
    test_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for digit in test_digits:
        labels = torch.tensor([digit] * 16).to(device)
        samples = ddpm.sample(labels, ddim_steps=ddim_steps)
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)

        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, output_dir / f'digit_{digit}_ddim{ddim_steps}.png')
        logger.info(f"  Generated digit {digit}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MNIST Digit Generation with DDIM')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--checkpoint', type=str, default='outputs/mnist_ddpm/final_model.pt',
                       help='Checkpoint path for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--ddim-steps', type=int, default=50,
                       help='Number of steps for DDIM sampling (default: 50)')

    args = parser.parse_args()

    if args.mode == 'train':
        logger.info("=== Training MNIST DDPM ===")
        train_mnist_ddpm(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        logger.info("=== Testing MNIST Generation (DDIM) ===")
        test_mnist_generation(
            checkpoint_path=args.checkpoint,
            device=args.device,
            ddim_steps=args.ddim_steps
        )

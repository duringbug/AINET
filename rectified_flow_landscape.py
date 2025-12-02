"""
Rectified Flow for Landscape Image Generation
Implements Rectified Flow (Reflow) for unconditional image generation.
Rectified Flow learns straight paths between noise and data distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
from pathlib import Path
import time

# Import components from ddpm_landscape
from ddpm_landscape import (
    LandscapeDataset,
    UNet,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RectifiedFlow(nn.Module):
    """
    Rectified Flow: Learning straight paths between noise and data.

    Key differences from DDPM:
    - Uses straight line paths: z_t = t * x_1 + (1-t) * x_0
    - Learns velocity field v_t instead of noise
    - Deterministic sampling (ODE-based)
    - Much faster sampling (can use 10-50 steps)
    """

    def __init__(
        self,
        model,
        device='cuda'
    ):
        super().__init__()

        self.model = model
        self.device = device

    def get_train_tuple(self, x1):
        """
        Generate training tuple (z_t, t, target_velocity)

        Args:
            x1: real data samples (batch of images)

        Returns:
            z_t: interpolated samples at time t
            t: random time steps
            v_target: target velocity (x1 - x0)
        """
        batch_size = x1.shape[0]

        # Sample random time steps uniformly from [0, 1]
        t = torch.rand(batch_size, device=self.device)

        # Sample noise (starting point x0)
        x0 = torch.randn_like(x1)

        # Interpolate: z_t = (1-t) * x0 + t * x1
        # This creates a straight line from x0 to x1
        t_expanded = t[:, None, None, None]
        z_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Target velocity: v = x1 - x0 (direction from x0 to x1)
        v_target = x1 - x0

        return z_t, t, v_target

    def forward(self, x1):
        """
        Training forward pass

        Args:
            x1: real data samples

        Returns:
            loss: MSE loss between predicted and target velocity
        """
        # Get training tuple
        z_t, t, v_target = self.get_train_tuple(x1)

        # Predict velocity at time t
        v_pred = self.model(z_t, t)

        # Loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample_ode(self, batch_size, channels=3, image_size=64, num_steps=50, method='euler'):
        """
        Sample images using ODE integration

        Args:
            batch_size: number of images to generate
            channels: number of channels (3 for RGB)
            image_size: size of generated images
            num_steps: number of integration steps
            method: integration method ('euler' or 'rk4')

        Returns:
            Generated images
        """
        self.model.eval()

        # Start from noise (t=0)
        z = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        # Time steps from 0 to 1
        dt = 1.0 / num_steps

        for i in tqdm(range(num_steps), desc=f'Sampling ({method})'):
            t = torch.full((batch_size,), i * dt, device=self.device)

            if method == 'euler':
                # Euler method: z_{t+dt} = z_t + v_t * dt
                v = self.model(z, t)
                z = z + v * dt

            elif method == 'rk4':
                # Runge-Kutta 4th order for better accuracy
                k1 = self.model(z, t)
                k2 = self.model(z + k1 * dt / 2, t + dt / 2)
                k3 = self.model(z + k2 * dt / 2, t + dt / 2)
                k4 = self.model(z + k3 * dt, t + dt)

                z = z + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

            else:
                raise ValueError(f"Unknown method: {method}")

        return z

    @torch.no_grad()
    def sample_ddim_style(self, batch_size, channels=3, image_size=64, num_steps=50):
        """
        Sample using DDIM-style discrete steps (alternative sampling method)

        This is similar to DDIM but adapted for Rectified Flow.
        """
        self.model.eval()

        # Start from noise
        z = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        # Create time steps
        timesteps = torch.linspace(0, 1, num_steps + 1).to(self.device)

        for i in tqdm(range(num_steps), desc='Sampling (DDIM-style)'):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((batch_size,), t_curr, device=self.device)

            # Predict velocity
            v = self.model(z, t_batch)

            # Update: z_{t+1} = z_t + v * (t_{next} - t_curr)
            z = z + v * (t_next - t_curr)

        return z


class TimeConditionedUNet(nn.Module):
    """
    Wrapper around UNet to handle time conditioning differently for Rectified Flow.
    In Rectified Flow, time t is in [0, 1] instead of discrete timesteps.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        """
        Args:
            x: input images
            t: continuous time in [0, 1]
        """
        # Convert continuous time [0, 1] to timestep-like values
        # Scale to a reasonable range (e.g., 0-999) for embedding
        timestep = (t * 999).long()

        return self.unet(x, timestep)


def train_rectified_flow(
    rectified_flow,
    train_loader,
    num_epochs=200,
    learning_rate=2e-4,
    save_dir='outputs/rectified_flow_landscape',
    sample_freq=10,
    save_freq=5,
    test_during_training=True,
    device='cuda',
    image_size=64
):
    """Train the Rectified Flow model

    Args:
        rectified_flow: the RectifiedFlow model
        train_loader: training data loader
        num_epochs: number of training epochs
        learning_rate: learning rate
        save_dir: directory to save checkpoints and samples
        sample_freq: frequency to generate samples (in epochs)
        save_freq: frequency to save checkpoints (in epochs)
        test_during_training: whether to test sampling during training
        device: device to use
        image_size: image size
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)
    (save_dir / 'training_tests').mkdir(exist_ok=True)

    optimizer = torch.optim.AdamW(
        rectified_flow.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    logger.info(f"Starting Rectified Flow training for {num_epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(num_epochs):
        rectified_flow.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, _ in pbar:
            images = images.to(device)
            # Normalize to [-1, 1]
            images = (images - 0.5) * 2.0

            # Compute loss
            loss = rectified_flow(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rectified_flow.model.parameters(), 1.0)
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

            # Sample with Euler method
            samples_euler = rectified_flow.sample_ode(
                batch_size=16,
                channels=3,
                image_size=image_size,
                num_steps=50,
                method='euler'
            )

            # Normalize to [0, 1]
            samples_euler = (samples_euler + 1.0) / 2.0
            samples_euler = torch.clamp(samples_euler, 0.0, 1.0)

            grid = make_grid(samples_euler, nrow=4, padding=2)
            save_image(grid, save_dir / 'samples' / f'sample_epoch_{epoch+1}_euler.png')

            # Test with different step counts during training
            if test_during_training and (epoch + 1) % (sample_freq * 2) == 0:
                logger.info(f'Running sampling tests with different step counts...')
                test_dir = save_dir / 'training_tests' / f'epoch_{epoch+1}'
                test_dir.mkdir(exist_ok=True, parents=True)

                for test_steps in [10, 25, 50]:
                    start_time = time.time()

                    samples = rectified_flow.sample_ode(
                        batch_size=16,
                        channels=3,
                        image_size=image_size,
                        num_steps=test_steps,
                        method='euler'
                    )

                    elapsed = time.time() - start_time

                    samples = (samples + 1.0) / 2.0
                    samples = torch.clamp(samples, 0.0, 1.0)

                    grid = make_grid(samples, nrow=4, padding=2)
                    save_image(grid, test_dir / f'{test_steps}steps_{elapsed:.2f}s.png')

                logger.info(f'Test samples saved to {test_dir}')

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_dir / 'checkpoints' / f'reflow_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': rectified_flow.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

        # Also save latest checkpoint every epoch for quick resume
        latest_path = save_dir / 'checkpoints' / 'latest.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': rectified_flow.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, latest_path)

    final_path = save_dir / 'reflow_final.pt'
    torch.save({'model_state_dict': rectified_flow.model.state_dict()}, final_path)
    logger.info(f'Final model saved to {final_path}')

    return rectified_flow


def test_rectified_flow(
    checkpoint_path='outputs/rectified_flow_landscape/reflow_final.pt',
    save_dir='outputs/rectified_flow_landscape/test_samples',
    num_samples=16,
    num_steps=50,
    image_size=64,
    device='cuda'
):
    """Test Rectified Flow sampling with a trained model"""

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load model
    logger.info(f'Loading model from {checkpoint_path}...')
    base_unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        use_attention=(False, True, True, False),
    ).to(device)

    model = TimeConditionedUNet(base_unet)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    rectified_flow = RectifiedFlow(model=model, device=device)

    logger.info(f'Model loaded successfully!')

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Test with different number of steps
    import time

    for steps in [10, 25, 50, 100]:
        logger.info(f'\n=== Testing with {steps} steps ===')

        # Euler method
        logger.info(f'Generating {num_samples} samples with Euler method...')
        start_time = time.time()

        samples = rectified_flow.sample_ode(
            batch_size=num_samples,
            channels=3,
            image_size=image_size,
            num_steps=steps,
            method='euler'
        )

        elapsed_time = time.time() - start_time
        logger.info(f'Time: {elapsed_time:.2f}s ({elapsed_time/num_samples:.3f}s per image)')

        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)

        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, save_dir / f'euler_{steps}steps.png')
        logger.info(f'Saved to {save_dir / f"euler_{steps}steps.png"}')

        # RK4 method (more accurate but slower)
        if steps <= 50:  # Only test RK4 for reasonable step counts
            logger.info(f'Generating {num_samples} samples with RK4 method...')
            start_time = time.time()

            samples = rectified_flow.sample_ode(
                batch_size=num_samples,
                channels=3,
                image_size=image_size,
                num_steps=steps,
                method='rk4'
            )

            elapsed_time = time.time() - start_time
            logger.info(f'Time: {elapsed_time:.2f}s ({elapsed_time/num_samples:.3f}s per image)')

            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            grid = make_grid(samples, nrow=4, padding=2)
            save_image(grid, save_dir / f'rk4_{steps}steps.png')
            logger.info(f'Saved to {save_dir / f"rk4_{steps}steps.png"}')

    logger.info(f'\nAll samples saved to {save_dir}')


def main(resume_from=None):
    """Main training function

    Args:
        resume_from: path to checkpoint to resume from (optional)
    """

    config = {
        'data_dir': 'data/landscape',
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 2e-4,
        'image_size': 64,
        'save_dir': 'outputs/rectified_flow_landscape',
        'sample_freq': 10,
        'save_freq': 5,  # Save checkpoints every 5 epochs
        'test_during_training': True,  # Test sampling during training
        'base_channels': 128,
        'channel_mult': (1, 2, 2, 4),
        'num_res_blocks': 2,
        'use_attention': (False, True, True, False),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('='*60)
    logger.info('RECTIFIED FLOW FOR LANDSCAPE GENERATION')
    logger.info('='*60)

    if resume_from:
        logger.info(f'Resuming from checkpoint: {resume_from}')

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
        return

    # Create model
    logger.info('Creating Rectified Flow model...')
    base_unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        base_channels=config['base_channels'],
        channel_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
    ).to(device)

    # Wrap with time conditioning
    model = TimeConditionedUNet(base_unet)

    rectified_flow = RectifiedFlow(model=model, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {total_params:,} ({trainable_params:,} trainable)')

    # Train model
    logger.info('Starting training...')
    logger.info(f'Training objective: Learn velocity field v(z_t, t) = x_1 - x_0')
    logger.info(f'Sampling method: ODE integration (Euler/RK4)')
    logger.info('='*60)

    train_rectified_flow(
        rectified_flow=rectified_flow,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        sample_freq=config['sample_freq'],
        save_freq=config['save_freq'],
        test_during_training=config['test_during_training'],
        device=device,
        image_size=config['image_size']
    )

    logger.info('Training completed!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rectified Flow for Landscape Images')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for testing or resuming training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='Number of sampling steps (for test mode)')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate (for test mode)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    if args.mode == 'train':
        main(resume_from=args.resume)
    elif args.mode == 'test':
        checkpoint = args.checkpoint or 'outputs/rectified_flow_landscape/reflow_final.pt'
        test_rectified_flow(
            checkpoint_path=checkpoint,
            num_steps=args.num_steps,
            num_samples=args.num_samples,
            device=args.device
        )

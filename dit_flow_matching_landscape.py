"""
Diffusion Transformer (DiT) + Flow Matching for Landscape Generation
Implements DiT architecture with Flow Matching training.
Supports variable resolution image generation.

Key Features:
- Transformer-based diffusion model (DiT)
- Flow Matching training (learns straight paths)
- Variable resolution support via position embedding interpolation
- Patch-based image processing (like ViT)
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
import math

# Import dataset from ddpm_landscape
from ddpm_landscape import LandscapeDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    """Modulation function for AdaLN"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    Converts images into patches and embeds them.
    """
    def __init__(self, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D) where N = H*W / patch_size^2
        """
        B, C, H, W = x.shape
        # Project and flatten
        x = self.proj(x)  # (B, D, H', W') where H' = H/patch_size
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, N, D) input tokens
        c: (B, D) conditioning (timestep embedding)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention with modulation
        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with modulation
        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for image generation.
    Supports variable resolution through position embedding interpolation.
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size

        # Patch embedding
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Position embedding (learnable, will be interpolated for different sizes)
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize pos_embed
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x, h, w):
        """
        x: (B, N, patch_size^2 * C)
        returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def get_pos_embed(self, h, w):
        """
        Get position embeddings with interpolation for variable resolution.
        h, w: number of patches in height and width
        """
        # Original size
        orig_size = int(self.pos_embed.shape[1] ** 0.5)

        if h == orig_size and w == orig_size:
            return self.pos_embed

        # Interpolate position embeddings
        pos_embed = self.pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pos_embed

    def forward(self, x, t):
        """
        x: (B, C, H, W) input images (can be variable size)
        t: (B,) timesteps (continuous values in [0, 1] for flow matching)
        """
        B, C, H, W = x.shape

        # Convert to patches
        x = self.x_embedder(x)  # (B, N, D)

        # Get position embeddings (with interpolation if needed)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        pos_embed = self.get_pos_embed(h_patches, w_patches)

        # Add position embeddings
        x = x + pos_embed

        # Get timestep embeddings
        t_emb = self.t_embedder(t)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final layer
        x = self.final_layer(x, t_emb)

        # Unpatchify
        x = self.unpatchify(x, h_patches, w_patches)

        return x


class FlowMatching(nn.Module):
    """
    Flow Matching with DiT backbone.
    Learns straight paths between noise and data.
    """
    def __init__(self, model, device='cuda'):
        super().__init__()
        self.model = model
        self.device = device

    def get_train_tuple(self, x1):
        """
        Generate training tuple for flow matching.

        Args:
            x1: real data samples

        Returns:
            z_t: interpolated samples at time t
            t: random time steps
            v_target: target velocity (x1 - x0)
        """
        batch_size = x1.shape[0]

        # Sample random time steps from [0, 1]
        t = torch.rand(batch_size, device=self.device)

        # Sample noise (starting point)
        x0 = torch.randn_like(x1)

        # Interpolate: z_t = (1-t) * x0 + t * x1
        t_expanded = t[:, None, None, None]
        z_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Target velocity: v = x1 - x0
        v_target = x1 - x0

        return z_t, t, v_target

    def forward(self, x1):
        """Training forward pass"""
        z_t, t, v_target = self.get_train_tuple(x1)

        # Predict velocity
        v_pred = self.model(z_t, t)

        # Loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, height=64, width=64, num_steps=50, method='euler', cfg_scale=0.0):
        """
        Sample images using ODE integration.

        Args:
            batch_size: number of images
            height: image height (can be different from training)
            width: image width (can be different from training)
            num_steps: number of integration steps
            method: 'euler' or 'rk4'
            cfg_scale: classifier-free guidance scale (not implemented yet)
        """
        self.model.eval()

        # Start from noise
        z = torch.randn(batch_size, 3, height, width).to(self.device)

        dt = 1.0 / num_steps

        for i in tqdm(range(num_steps), desc=f'Sampling ({method})'):
            t = torch.full((batch_size,), i * dt, device=self.device)

            if method == 'euler':
                v = self.model(z, t)
                z = z + v * dt

            elif method == 'rk4':
                k1 = self.model(z, t)
                k2 = self.model(z + k1 * dt / 2, t + dt / 2)
                k3 = self.model(z + k2 * dt / 2, t + dt / 2)
                k4 = self.model(z + k3 * dt, t + dt)

                z = z + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

            else:
                raise ValueError(f"Unknown method: {method}")

        return z


class VariableResolutionDataset(LandscapeDataset):
    """
    Dataset that returns images at variable resolutions.
    Useful for training DiT with multiple resolutions.
    """
    def __init__(self, data_dir, image_sizes=[64, 96, 128], transform=None):
        # Don't set a fixed image_size in parent
        super().__init__(data_dir, image_size=64, transform=None)

        self.image_sizes = image_sizes

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms

        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            # Randomly choose a size
            size = self.image_sizes[torch.randint(0, len(self.image_sizes), (1,)).item()]

            # Transform
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            image = transform(image)
            return image, 0

        except Exception as e:
            logger.warning(f'Error loading image {img_path}: {e}')
            size = self.image_sizes[0]
            return torch.randn(3, size, size), 0


def train_dit_flow(
    flow_matching,
    train_loader,
    num_epochs=200,
    learning_rate=1e-4,
    save_dir='outputs/dit_flow_landscape',
    sample_freq=10,
    save_freq=5,
    test_during_training=True,
    device='cuda',
    test_sizes=[(64, 64), (96, 96), (128, 128)]
):
    """Train DiT with Flow Matching"""

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)
    (save_dir / 'training_tests').mkdir(exist_ok=True)

    optimizer = torch.optim.AdamW(
        flow_matching.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    logger.info(f"Starting DiT + Flow Matching training for {num_epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(num_epochs):
        flow_matching.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, _ in pbar:
            images = images.to(device)
            # Normalize to [-1, 1]
            images = (images - 0.5) * 2.0

            # Compute loss
            loss = flow_matching(images)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_matching.model.parameters(), 1.0)
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

            # Sample at default resolution
            samples = flow_matching.sample(
                batch_size=16,
                height=64,
                width=64,
                num_steps=50,
                method='euler'
            )

            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            grid = make_grid(samples, nrow=4, padding=2)
            save_image(grid, save_dir / 'samples' / f'sample_epoch_{epoch+1}.png')

            # Test variable resolutions
            if test_during_training and (epoch + 1) % (sample_freq * 2) == 0:
                logger.info(f'Testing variable resolutions...')
                test_dir = save_dir / 'training_tests' / f'epoch_{epoch+1}'
                test_dir.mkdir(exist_ok=True, parents=True)

                for h, w in test_sizes:
                    logger.info(f'  Generating {h}x{w} images...')
                    start_time = time.time()

                    samples = flow_matching.sample(
                        batch_size=16,
                        height=h,
                        width=w,
                        num_steps=50,
                        method='euler'
                    )

                    elapsed = time.time() - start_time

                    samples = (samples + 1.0) / 2.0
                    samples = torch.clamp(samples, 0.0, 1.0)

                    grid = make_grid(samples, nrow=4, padding=2)
                    save_image(grid, test_dir / f'{h}x{w}_{elapsed:.2f}s.png')

                logger.info(f'Variable resolution samples saved to {test_dir}')

        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_dir / 'checkpoints' / f'dit_flow_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': flow_matching.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

        # Save latest checkpoint
        latest_path = save_dir / 'checkpoints' / 'latest.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': flow_matching.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, latest_path)

    # Save final model
    final_path = save_dir / 'dit_flow_final.pt'
    torch.save({'model_state_dict': flow_matching.model.state_dict()}, final_path)
    logger.info(f'Final model saved to {final_path}')

    return flow_matching


def test_dit_flow(
    checkpoint_path='outputs/dit_flow_landscape/dit_flow_final.pt',
    save_dir='outputs/dit_flow_landscape/test_samples',
    num_samples=16,
    test_resolutions=[(64, 64), (96, 96), (128, 128), (64, 128), (128, 64)],
    num_steps=50,
    device='cuda'
):
    """Test DiT Flow Matching with variable resolutions"""

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load model
    logger.info(f'Loading model from {checkpoint_path}...')
    dit = DiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    dit.load_state_dict(checkpoint['model_state_dict'])

    flow_matching = FlowMatching(model=dit, device=device)

    logger.info(f'Model loaded successfully!')

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Test different resolutions
    for h, w in test_resolutions:
        logger.info(f'\n=== Generating {h}x{w} images ===')

        start_time = time.time()

        samples = flow_matching.sample(
            batch_size=num_samples,
            height=h,
            width=w,
            num_steps=num_steps,
            method='euler'
        )

        elapsed = time.time() - start_time
        logger.info(f'Time: {elapsed:.2f}s ({elapsed/num_samples:.3f}s per image)')

        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)

        grid = make_grid(samples, nrow=4, padding=2)
        save_image(grid, save_dir / f'{h}x{w}_steps{num_steps}.png')
        logger.info(f'Saved to {save_dir / f"{h}x{w}_steps{num_steps}.png"}')

    logger.info(f'\nAll samples saved to {save_dir}')


def main():
    """Main training function"""

    config = {
        'data_dir': 'data/landscape',
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'save_dir': 'outputs/dit_flow_landscape',
        'sample_freq': 10,
        'save_freq': 5,
        'test_during_training': True,

        # DiT config
        'input_size': 64,  # Base training size
        'patch_size': 4,
        'hidden_size': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,

        # Variable resolution
        'train_sizes': [64],  # Can use [64, 96, 128] for multi-res training
        'test_sizes': [(64, 64), (96, 96), (128, 128)],
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('='*60)
    logger.info('DIFFUSION TRANSFORMER + FLOW MATCHING')
    logger.info('Variable Resolution Landscape Generation')
    logger.info('='*60)

    # Load dataset
    logger.info(f'Loading landscape images from {config["data_dir"]}...')

    try:
        if len(config['train_sizes']) > 1:
            # Variable resolution training
            train_dataset = VariableResolutionDataset(
                config['data_dir'],
                image_sizes=config['train_sizes']
            )
        else:
            # Fixed resolution training
            train_dataset = LandscapeDataset(
                config['data_dir'],
                image_size=config['train_sizes'][0]
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
        logger.info(f'Training resolutions: {config["train_sizes"]}')

    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        logger.info('Please make sure you have images in the data/landscape directory')
        return

    # Create model
    logger.info('Creating DiT model...')
    dit = DiT(
        input_size=config['input_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
    ).to(device)

    flow_matching = FlowMatching(model=dit, device=device)

    total_params = sum(p.numel() for p in dit.parameters())
    trainable_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {total_params:,} ({trainable_params:,} trainable)')

    # Train model
    logger.info('Starting training...')
    logger.info(f'Architecture: DiT (Diffusion Transformer)')
    logger.info(f'Training: Flow Matching (straight paths)')
    logger.info(f'Patch size: {config["patch_size"]}x{config["patch_size"]}')
    logger.info(f'Position embedding: Interpolated for variable resolution')
    logger.info('='*60)

    train_dit_flow(
        flow_matching=flow_matching,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        sample_freq=config['sample_freq'],
        save_freq=config['save_freq'],
        test_during_training=config['test_during_training'],
        device=device,
        test_sizes=config['test_sizes']
    )

    logger.info('Training completed!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DiT + Flow Matching for Landscape Images')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for testing')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='Number of sampling steps')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--height', type=int, default=64,
                       help='Image height')
    parser.add_argument('--width', type=int, default=64,
                       help='Image width')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        checkpoint = args.checkpoint or 'outputs/dit_flow_landscape/dit_flow_final.pt'

        # Test with custom resolution or default resolutions
        if args.height != 64 or args.width != 64:
            test_resolutions = [(args.height, args.width)]
        else:
            test_resolutions = [(64, 64), (96, 96), (128, 128), (64, 128), (128, 64)]

        test_dit_flow(
            checkpoint_path=checkpoint,
            num_samples=args.num_samples,
            test_resolutions=test_resolutions,
            num_steps=args.num_steps,
            device=args.device
        )

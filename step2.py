"""
Step 2: Text-Conditioned Image Diffusion Model

训练一个文本条件的图像扩散模型，实现真正的文图映射：
1. 从 main.py 加载预训练的文本编码器
2. 基于 ddpm_landscape.py 的 UNet，添加文本条件
3. 训练文本条件的 DDPM
4. 生成：文本 → 文本编码 → 条件扩散 → 图像

架构流程：
Step 1 (main.py): 训练文本/图像编码器对齐
Step 2 (本文件): 训练文本条件的图像扩散模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
from pathlib import Path

# Import from main.py
from main import (
    Flickr30kDataset,
    SimpleTokenizer,
    TextEncoder,
)

# Import from ddpm_landscape.py
from ddpm_landscape import (
    SinusoidalPositionEmbeddings,
    AttentionBlock,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning on text embeddings - Memory efficient version"""

    def __init__(self, query_dim, context_dim, num_heads=4):  # Reduced default heads
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.head_dim = query_dim // num_heads

        self.scale = self.head_dim ** -0.5

        # Query from image features
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        # Key and value from text embeddings
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)

        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: (batch_size, seq_len, query_dim) - image features
            context: (batch_size, context_len, context_dim) - text embeddings
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.to_q(x)  # (batch_size, seq_len, query_dim)
        k = self.to_k(context)  # (batch_size, context_len, query_dim)
        v = self.to_v(context)  # (batch_size, context_len, query_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Use scaled_dot_product_attention if available (more memory efficient)
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            # Fallback to manual attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.query_dim)
        out = self.to_out(out)

        return out


class ConditionedBlock(nn.Module):
    """
    Convolutional block with text conditioning
    基于 ddpm_landscape.py 的 Block，添加文本交叉注意力
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Text conditioning via cross-attention
        # First, we need to reshape spatial features to sequence
        # Then apply cross-attention with text
        self.text_cross_attn = CrossAttention(
            query_dim=out_channels,
            context_dim=text_emb_dim,
            num_heads=min(4, max(1, out_channels // 32))  # Reduced for memory efficiency
        )
        self.text_norm = nn.GroupNorm(groups, out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb, text_context):
        """
        Args:
            x: (batch_size, in_channels, H, W)
            time_emb: (batch_size, time_emb_dim)
            text_context: (batch_size, text_seq_len, text_emb_dim)
        """
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

        # Text cross-attention
        # Reshape (B, C, H, W) -> (B, H*W, C)
        batch_size, channels, height, width = h.shape
        h_seq = h.view(batch_size, channels, height * width).transpose(1, 2)

        # Apply cross-attention
        h_attn = self.text_cross_attn(h_seq, text_context)

        # Reshape back (B, H*W, C) -> (B, C, H, W)
        h_attn = h_attn.transpose(1, 2).view(batch_size, channels, height, width)
        h = h + h_attn
        h = self.text_norm(h)

        # Residual connection
        return h + self.residual_conv(x)


class TextConditionedUNet(nn.Module):
    """
    Text-conditioned UNet for DDPM
    基于 ddpm_landscape.py 的 UNet，添加文本条件
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=256,
        text_emb_dim=256,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        use_attention=(False, True, True, False),
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim
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

            # Residual blocks with text conditioning
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ConditionedBlock(ch_in, ch_out, time_emb_dim, text_emb_dim))
                ch_in = ch_out

            self.down_blocks.append(blocks)

            # Self-attention
            if use_attention[i]:
                self.down_attentions.append(AttentionBlock(ch_out))
            else:
                self.down_attentions.append(nn.Identity())

            # Downsample
            if i < len(channel_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(ch_out, ch_out, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.down_samples.append(nn.Identity())

        # Bottleneck
        bottleneck_channels = base_channels * channel_mult[-1]
        self.bottleneck = nn.ModuleList([
            ConditionedBlock(bottleneck_channels, bottleneck_channels, time_emb_dim, text_emb_dim),
            AttentionBlock(bottleneck_channels),
            ConditionedBlock(bottleneck_channels, bottleneck_channels, time_emb_dim, text_emb_dim),
        ])

        # Decoder
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            ch_out = base_channels * mult
            ch_in_skip = base_channels * channel_mult[-(i + 1)]

            # Upsample
            if i > 0:
                ch_in_current = base_channels * channel_mult[-(i)]
                self.up_samples.append(
                    nn.ConvTranspose2d(ch_in_current, ch_in_current, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())

            # Residual blocks with text conditioning
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                if j == 0:
                    if i > 0:
                        ch_in_current = base_channels * channel_mult[-(i)]
                    else:
                        ch_in_current = base_channels * channel_mult[-1]
                    blocks.append(ConditionedBlock(ch_in_current + ch_in_skip, ch_out, time_emb_dim, text_emb_dim))
                else:
                    blocks.append(ConditionedBlock(ch_out, ch_out, time_emb_dim, text_emb_dim))

            self.up_blocks.append(blocks)

            # Self-attention
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

    def forward(self, x, timestep, text_context):
        """
        Args:
            x: (batch_size, 3, H, W) - noisy image
            timestep: (batch_size,) - diffusion timestep
            text_context: (batch_size, seq_len, text_emb_dim) - text embeddings
        """
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial convolution
        x = self.init_conv(x)

        # Encoder
        skips = []
        for blocks, attn, downsample in zip(self.down_blocks, self.down_attentions, self.down_samples):
            for block in blocks:
                x = block(x, t, text_context)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck[0](x, t, text_context)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t, text_context)

        # Decoder
        for blocks, attn, upsample, skip in zip(
            self.up_blocks, self.up_attentions, self.up_samples, reversed(skips)
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)

            for block in blocks:
                x = block(x, t, text_context)

            x = attn(x)

        # Final convolution
        x = self.final_conv(x)

        return x


class TextConditionedDDPM(nn.Module):
    """
    Text-conditioned DDPM
    基于 ddpm_landscape.py 的 DDPM，添加文本条件
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

        # Create beta schedule
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
        """Add noise (forward diffusion)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean_variance(self, x_t, t, text_context):
        """Predict mean and variance"""
        predicted_noise = self.model(x_t, t, text_context)

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
    def p_sample(self, x_t, t, text_context):
        """Sample one step"""
        batch_size = x_t.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, text_context)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(batch_size, 1, 1, 1)

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(self, text_context, channels=3, image_size=64):
        """
        Generate images conditioned on text

        Args:
            text_context: (batch_size, seq_len, text_emb_dim) - text embeddings
            channels: number of channels
            image_size: size of generated images
        """
        self.model.eval()
        batch_size = text_context.shape[0]

        # Start from random noise
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)

        # Reverse diffusion process
        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
            x = self.p_sample(x, t_batch, text_context)

        return x

    def forward(self, x_start, text_context):
        """
        Training forward pass

        Args:
            x_start: (batch_size, 3, H, W) - clean images
            text_context: (batch_size, seq_len, text_emb_dim) - text embeddings
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(self.device)
        noise = torch.randn_like(x_start)

        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        predicted_noise = self.model(x_noisy, t, text_context)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss


def train_text_conditioned_ddpm(
    ddpm,
    text_encoder,
    train_loader,
    num_epochs=100,
    learning_rate=2e-4,
    save_dir='outputs/text_conditioned_ddpm',
    sample_freq=10,
    device='cuda',
    image_size=64,
    freeze_text_encoder=True,
    gradient_accumulation_steps=1,
    use_mixed_precision=False
):
    """
    Train text-conditioned DDPM

    Args:
        ddpm: TextConditionedDDPM model
        text_encoder: Pretrained text encoder from main.py
        train_loader: DataLoader for image-text pairs
        freeze_text_encoder: Whether to freeze text encoder
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)

    # Freeze text encoder if specified
    if freeze_text_encoder:
        for param in text_encoder.parameters():
            param.requires_grad = False
        text_encoder.eval()
        logger.info("Text encoder frozen")
    else:
        logger.info("Text encoder will be fine-tuned")

    # Optimizer
    if freeze_text_encoder:
        optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            list(ddpm.model.parameters()) + list(text_encoder.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    if use_mixed_precision:
        logger.info("Using mixed precision training (FP16)")

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Total batches per epoch: {len(train_loader)}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")

    for epoch in range(num_epochs):
        ddpm.model.train()
        if not freeze_text_encoder:
            text_encoder.train()

        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)

            # Normalize images to [-1, 1]
            images = (images - 0.5) * 2.0

            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Encode text
                with torch.set_grad_enabled(not freeze_text_encoder):
                    text_embeddings = text_encoder(tokens)  # (batch_size, text_emb_dim)
                    # Add sequence dimension: (batch_size, 1, text_emb_dim)
                    text_context = text_embeddings.unsqueeze(1)

                # DDPM loss
                loss = ddpm(images, text_context)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backward
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
                    if not freeze_text_encoder:
                        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
                    if not freeze_text_encoder:
                        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}')

        scheduler.step()

        # Generate samples
        if (epoch + 1) % sample_freq == 0:
            logger.info(f'Generating samples at epoch {epoch+1}...')

            # Get some random captions from dataset
            num_samples = min(16, len(train_loader.dataset))
            indices = torch.randperm(len(train_loader.dataset))[:num_samples]

            captions = []
            tokens_list = []

            for idx in indices:
                sample = train_loader.dataset[idx.item()]
                captions.append(sample['caption'])
                tokens_list.append(sample['tokens'])

            tokens = torch.stack(tokens_list).to(device)

            # Encode text
            with torch.no_grad():
                text_encoder.eval()
                text_embeddings = text_encoder(tokens)
                text_context = text_embeddings.unsqueeze(1)

            # Generate images
            samples = ddpm.sample(text_context, channels=3, image_size=image_size)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

            grid = make_grid(samples, nrow=4, padding=2)
            save_image(grid, save_dir / 'samples' / f'sample_epoch_{epoch+1}.png')

            # Save captions
            with open(save_dir / 'samples' / f'captions_epoch_{epoch+1}.txt', 'w') as f:
                for i, caption in enumerate(captions):
                    f.write(f"{i+1}. {caption}\n")

        # Save checkpoint
        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm.model.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')

    final_path = save_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': ddpm.model.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
    }, final_path)
    logger.info(f'Final model saved to {final_path}')

    return ddpm


def main():
    """Main training function"""

    config = {
        # Data config
        'data_dir': 'data/flickr30k',
        'batch_size': 4,  # Reduced for memory efficiency
        'gradient_accumulation_steps': 4,  # Effective batch size = 4 * 4 = 16
        'num_epochs': 100,
        'learning_rate': 2e-4,
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'image_size': 64,  # Start with 64, can increase to 128 or 256
        'save_dir': 'outputs/text_conditioned_ddpm',
        'sample_freq': 10,

        # Model config - Reduced for memory efficiency
        'vocab_size': 10000,
        'text_emb_dim': 256,
        'time_emb_dim': 256,
        'base_channels': 64,  # Reduced from 128
        'channel_mult': (1, 2, 2),  # Reduced from (1, 2, 2, 4)
        'num_res_blocks': 1,  # Reduced from 2
        'use_attention': (False, True, True),  # Adjusted for 3 levels

        # Pretrained model config
        'pretrained_checkpoint': 'outputs/best_model/pytorch_model.bin',  # From main.py
        'freeze_text_encoder': True,  # Set to False to fine-tune text encoder

        # Memory optimization
        'use_mixed_precision': True,  # Enable automatic mixed precision
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Clear CUDA cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total')

    # Load pretrained text encoder from main.py
    logger.info(f'Loading pretrained text encoder from {config["pretrained_checkpoint"]}...')
    try:
        checkpoint = torch.load(config['pretrained_checkpoint'], map_location=device)

        # Initialize tokenizer
        tokenizer = SimpleTokenizer(
            vocab_size=checkpoint['tokenizer']['vocab_size'],
            max_length=checkpoint['tokenizer']['max_length']
        )
        tokenizer.word2idx = checkpoint['tokenizer']['word2idx']
        tokenizer.idx2word = checkpoint['tokenizer']['idx2word']

        # Initialize text encoder
        text_encoder = TextEncoder(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=config['text_emb_dim']
        ).to(device)

        # Load pretrained weights
        text_encoder_state = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('text_encoder.'):
                new_key = key.replace('text_encoder.', '')
                text_encoder_state[new_key] = value

        text_encoder.load_state_dict(text_encoder_state, strict=False)
        logger.info("Pretrained text encoder loaded successfully!")

    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {config['pretrained_checkpoint']}")
        logger.info("Please train main.py first to get pretrained text encoder")
        logger.info("Or set pretrained_checkpoint to None to train from scratch")
        return
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        logger.info("Training text encoder from scratch...")

        # Initialize tokenizer and text encoder from scratch
        tokenizer = SimpleTokenizer(
            vocab_size=config['vocab_size'],
            max_length=77
        )

        # Build vocab from dataset
        temp_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            split='all'
        )
        tokenizer.build_vocab(temp_dataset.get_all_captions())

        text_encoder = TextEncoder(
            vocab_size=len(tokenizer.word2idx),
            embed_dim=config['text_emb_dim']
        ).to(device)

    # Load dataset
    logger.info(f'Loading dataset from {config["data_dir"]}...')
    try:
        train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            tokenizer=tokenizer,
            split='train',
            train_ratio=0.9
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
        logger.info('Please make sure you have the dataset in the correct location')
        return

    # Create text-conditioned UNet
    logger.info('Creating text-conditioned UNet...')
    unet = TextConditionedUNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config['time_emb_dim'],
        text_emb_dim=config['text_emb_dim'],
        base_channels=config['base_channels'],
        channel_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
    ).to(device)

    # Create text-conditioned DDPM
    ddpm = TextConditionedDDPM(
        model=unet,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f'UNet parameters: {total_params:,} ({trainable_params:,} trainable)')

    # Train model
    logger.info('Starting training...')
    train_text_conditioned_ddpm(
        ddpm=ddpm,
        text_encoder=text_encoder,
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        sample_freq=config['sample_freq'],
        device=device,
        image_size=config['image_size'],
        freeze_text_encoder=config['freeze_text_encoder'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        use_mixed_precision=config.get('use_mixed_precision', False)
    )

    logger.info('Training completed!')


if __name__ == '__main__':
    main()

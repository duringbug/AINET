"""
Stable Diffusion 1.5 + LoRA 微调（使用已微调的 CLIP LoRA）

架构：
- CLIP LoRA: 加载已微调的 checkpoint（冻结，用于文本编码）
- SD1.5 UNet: 注入 LoRA（训练）
- SD1.5 VAE: 冻结（用于编码/解码）

优势：
- 快速训练：只训练 UNet LoRA
- 强大文本理解：使用已适配 Flickr30k 的 CLIP LoRA
- 高质量生成：512x512 分辨率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
from pathlib import Path
import math
import matplotlib.pyplot as plt

# Diffusers for Stable Diffusion 1.5
try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        DDIMScheduler,
    )
    import clip
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'diffusers', 'transformers', 'accelerate'])
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        DDPMScheduler,
        DDIMScheduler,
    )
    import clip

# Import CLIP LoRA Trainer
from train_clip_lora import CLIPLoRATrainer

# Dataset
from main import Flickr30kDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Text Projection Layer
# ============================================================

class TextProjection(nn.Module):
    """Project CLIP 512-dim features to SD1.5 768-dim"""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(512, 768)
        nn.init.normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        # x: (batch, seq_len, 512)
        return self.projection(x)


# ============================================================
# LoRA Implementation (for UNet)
# ============================================================

class LoRALayer(nn.Module):
    """LoRA Layer for efficient fine-tuning"""
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices - stored in float32 for stability
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.float32))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=0.01)

    def forward(self, x):
        original_dtype = x.dtype
        x_fp32 = x.to(dtype=torch.float32)
        lora_output = (x_fp32 @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output.to(dtype=original_dtype)


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear: nn.Linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        device = next(linear.parameters()).device

        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha
        ).to(device=device)

        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def inject_lora_into_unet(unet: UNet2DConditionModel, rank=4, alpha=1.0):
    """在 UNet 的 attention 层中注入 LoRA"""
    lora_params = []
    injection_count = 0

    for name, module in unet.named_modules():
        if 'attn' in name and isinstance(module, nn.Linear):
            if any(x in name for x in ['to_q', 'to_k', 'to_v', 'to_out.0']):
                *parent_path, attr_name = name.split('.')
                parent = unet
                for p in parent_path:
                    parent = getattr(parent, p)

                original_linear = getattr(parent, attr_name)
                lora_linear = LinearWithLoRA(original_linear, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_linear)

                lora_params.extend(lora_linear.lora.parameters())
                injection_count += 1

                if injection_count <= 3:
                    logger.info(f"  ✓ {name}")

    logger.info(f"Injected LoRA into UNet: {injection_count} layers")
    return lora_params


# ============================================================
# Training Functions
# ============================================================

def plot_loss_curve(loss_history, save_path, window_size=100):
    """绘制训练损失曲线"""
    if len(loss_history) < 2:
        return

    plt.figure(figsize=(15, 5))

    # 左图：所有loss点 + 移动平均
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, alpha=0.3, label='Raw Loss', linewidth=0.5)
    moving_avg = []
    for i in range(len(loss_history)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(sum(loss_history[start_idx:i+1]) / (i - start_idx + 1))
    plt.plot(moving_avg, linewidth=2, label=f'Moving Avg ({window_size})', color='red')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss (All Batches)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 中图：最近的loss（放大）
    recent_steps = min(1000, len(loss_history))
    plt.subplot(1, 3, 2)
    plt.plot(range(len(loss_history) - recent_steps, len(loss_history)),
             loss_history[-recent_steps:], alpha=0.5, label='Raw Loss')
    plt.plot(range(len(loss_history) - recent_steps, len(loss_history)),
             moving_avg[-recent_steps:], linewidth=2, label=f'Moving Avg ({window_size})', color='red')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (Last {recent_steps} Batches)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：Loss分布（直方图）
    plt.subplot(1, 3, 3)
    plt.hist(loss_history, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.axvline(sum(loss_history) / len(loss_history), color='red', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def train_sd15_with_clip_lora(
    vae,
    clip_lora_trainer,
    text_projection,
    unet,
    noise_scheduler,
    train_loader,
    config,
    device='cuda'
):
    """
    训练 SD1.5 UNet（使用 CLIP LoRA 作为文本编码器）

    Args:
        vae: 预训练的 VAE (冻结)
        clip_lora_trainer: CLIP LoRA Trainer (冻结)
        text_projection: 512->768 projection layer (训练)
        unet: UNet + LoRA (训练)
        noise_scheduler: DDPM scheduler
        train_loader: DataLoader
        config: 配置字典
    """
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)

    # Freeze VAE and CLIP
    vae.requires_grad_(False)
    vae.eval()
    clip_lora_trainer.model.eval()

    # 收集 UNet LoRA 参数 + Text Projection 参数
    trainable_params = []
    for module in unet.modules():
        if isinstance(module, LinearWithLoRA):
            trainable_params.extend(module.lora.parameters())

    # Add text projection parameters
    trainable_params.extend(text_projection.parameters())

    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in unet.parameters())
    logger.info(f"Total UNet parameters: {num_total:,}")
    logger.info(f"Trainable LoRA parameters: {num_trainable:,} ({100 * num_trainable / num_total:.2f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay'],
        eps=1e-8
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=config['learning_rate'] * 0.1
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config['use_mixed_precision'] else None
    if config['use_mixed_precision']:
        logger.info("Using mixed precision training (FP16)")

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for {config['num_epochs']} epochs")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"{'='*60}\n")

    global_step = 0
    all_loss_history = []

    for epoch in range(config['num_epochs']):
        unet.train()
        clip_lora_trainer.model.eval()  # CLIP 始终 eval

        total_loss = 0
        min_loss = float('inf')
        max_loss = 0
        loss_history = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            captions = batch['caption']

            # Encode images to latent space
            with torch.no_grad():
                # Normalize to [-1, 1]
                images = images * 2.0 - 1.0
                # Convert to same dtype as VAE (FP16)
                images = images.to(dtype=vae.dtype)
                # Encode to latents
                latents = vae.encode(images).latent_dist.sample()
                # Scale latents
                latents = latents * vae.config.scaling_factor

            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text with CLIP LoRA
            with torch.amp.autocast('cuda', enabled=config['use_mixed_precision']):
                # 使用 CLIP 的 tokenizer
                with torch.no_grad():
                    text_tokens = clip.tokenize(captions, truncate=True).to(device)
                    # Get text embeddings from CLIP LoRA (512 dim)
                    text_features = clip_lora_trainer.model.encode_text(text_tokens)
                    # Normalize
                    text_features = F.normalize(text_features, dim=-1)
                    # Add sequence dimension: (batch, 512) -> (batch, 1, 512)
                    text_features = text_features.unsqueeze(1).float()

                # Project to 768 dim for SD1.5
                encoder_hidden_states = text_projection(text_features)

                # Predict noise with UNet
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Compute loss
                loss = F.mse_loss(model_pred, noise, reduction='mean')
                loss = loss / config['gradient_accumulation_steps']

            # Backward
            if config['use_mixed_precision']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                if config['use_mixed_precision']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config['max_grad_norm'])
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 记录loss
            current_loss = loss.item() * config['gradient_accumulation_steps']
            total_loss += current_loss
            min_loss = min(min_loss, current_loss)
            max_loss = max(max_loss, current_loss)

            loss_history.append(current_loss)
            all_loss_history.append(current_loss)

            # 移动平均
            window = loss_history[-100:] if len(loss_history) > 100 else loss_history
            moving_avg = sum(window) / len(window)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg': f'{moving_avg:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'step': global_step
            })

        avg_loss = total_loss / len(train_loader)
        logger.info(f'\n[Epoch {epoch+1}/{config["num_epochs"]}] '
                   f'Avg Loss: {avg_loss:.4f} | '
                   f'Min: {min_loss:.4f} | '
                   f'Max: {max_loss:.4f}\n')

        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 绘制loss曲线
        if len(all_loss_history) > 0:
            loss_plot_path = save_dir / 'loss_curve.png'
            plot_loss_curve(all_loss_history, loss_plot_path, window_size=100)

        # Generate samples
        if (epoch + 1) % config['sample_freq'] == 0:
            logger.info(f'Generating samples at epoch {epoch+1}...')
            generate_samples(
                vae, clip_lora_trainer, text_projection, unet,
                train_loader.dataset,
                config,
                epoch + 1,
                save_dir / 'samples',
                device
            )

        # Save checkpoint
        if (epoch + 1) % config['checkpoint_freq'] == 0:
            checkpoint_path = save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt'
            save_checkpoint(
                unet, text_projection, optimizer, lr_scheduler,
                epoch + 1, avg_loss, config, checkpoint_path, all_loss_history
            )
            logger.info(f'Checkpoint saved to {checkpoint_path}')

    # Save final model
    final_path = save_dir / 'final_model.pt'
    save_checkpoint(
        unet, text_projection, optimizer, lr_scheduler,
        config['num_epochs'], avg_loss, config, final_path, all_loss_history
    )
    logger.info(f'\nFinal model saved to {final_path}')

    # 保存loss历史
    loss_history_path = save_dir / 'loss_history.txt'
    with open(loss_history_path, 'w') as f:
        f.write('# Batch\tLoss\n')
        for i, loss_val in enumerate(all_loss_history):
            f.write(f'{i+1}\t{loss_val:.6f}\n')
    logger.info(f'Loss history saved to {loss_history_path}')


def save_checkpoint(unet, text_projection, optimizer, lr_scheduler, epoch, loss, config, path, loss_history=None):
    """保存检查点，保存 UNet LoRA + Text Projection 参数"""
    unet_lora_state = {}
    for name, module in unet.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_name = name + '.lora'
            unet_lora_state[lora_name] = module.lora.state_dict()

    checkpoint_dict = {
        'epoch': epoch,
        'unet_lora_state': unet_lora_state,
        'text_projection_state': text_projection.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
        'config': config,
    }

    if loss_history is not None:
        checkpoint_dict['loss_history'] = loss_history

    torch.save(checkpoint_dict, path)


def load_checkpoint(unet, text_projection, checkpoint_path, device='cuda'):
    """加载检查点，恢复 UNet LoRA + Text Projection 参数"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    unet_lora_state = checkpoint['unet_lora_state']
    for name, module in unet.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_name = name + '.lora'
            if lora_name in unet_lora_state:
                module.lora.load_state_dict(unet_lora_state[lora_name])

    # Load text projection
    if 'text_projection_state' in checkpoint:
        text_projection.load_state_dict(checkpoint['text_projection_state'])

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return checkpoint


@torch.no_grad()
def generate_samples(vae, clip_lora_trainer, text_projection, unet, dataset, config, epoch, save_dir, device):
    """生成样本图像"""
    vae.eval()
    clip_lora_trainer.model.eval()
    text_projection.eval()
    unet.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # 选择随机样本
    num_samples = min(8, len(dataset))
    indices = torch.randperm(len(dataset))[:num_samples]
    captions = [dataset[idx.item()]['caption'] for idx in indices]

    logger.info(f"Generating {num_samples} samples...")

    # DDIM scheduler for faster sampling
    scheduler = DDIMScheduler.from_pretrained(
        config['model_name'],
        subfolder='scheduler'
    )
    scheduler.set_timesteps(config['num_inference_steps'])

    # Encode text with CLIP LoRA
    text_tokens = clip.tokenize(captions, truncate=True).to(device)
    text_features = clip_lora_trainer.model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.unsqueeze(1).float()

    # Project to 768 dim
    encoder_hidden_states = text_projection(text_features)

    # Generate latents
    latents = torch.randn(
        (num_samples, 4, config['image_size'] // 8, config['image_size'] // 8),
        device=device
    )
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc='Sampling', leave=False):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    # Save grid
    grid = make_grid(images, nrow=4, padding=2)
    save_image(grid, save_dir / f'sample_epoch_{epoch}.png')

    # Save individual images
    for i, img in enumerate(images):
        save_image(img, save_dir / f'sample_epoch_{epoch}_img_{i+1}.png')

    # Save captions
    with open(save_dir / f'captions_epoch_{epoch}.txt', 'w') as f:
        for i, caption in enumerate(captions):
            f.write(f"{i+1}. {caption}\n")

    logger.info(f"Samples saved to {save_dir / f'sample_epoch_{epoch}.png'}")

    unet.train()


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def generate_images_from_prompts(
    clip_lora_checkpoint,
    unet_checkpoint,
    prompts,
    config,
    output_dir='outputs/sd15_clip_lora/generated',
    num_images_per_prompt=4,
    device='cuda'
):
    """从文本提示生成图像"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load CLIP LoRA
    logger.info(f'Loading CLIP LoRA from {clip_lora_checkpoint}...')
    clip_lora_trainer = CLIPLoRATrainer(
        model_name='ViT-B/32',
        rank=config['clip_lora_rank'],
        alpha=config['clip_lora_alpha'],
        device=device
    )
    clip_lora_trainer.load_checkpoint(clip_lora_checkpoint)
    clip_lora_trainer.model.eval()

    # Load SD1.5 components
    logger.info('Loading Stable Diffusion 1.5 models...')
    vae = AutoencoderKL.from_pretrained(
        config['model_name'],
        subfolder='vae',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        config['model_name'],
        subfolder='unet',
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)

    # Inject LoRA
    logger.info('Injecting LoRA into UNet...')
    inject_lora_into_unet(unet, rank=config['lora_rank'], alpha=config['lora_alpha'])

    # Create text projection
    text_projection = TextProjection().to(device)

    # Load checkpoint
    logger.info(f'Loading UNet checkpoint from {unet_checkpoint}...')
    load_checkpoint(unet, text_projection, unet_checkpoint, device)

    vae.eval()
    text_projection.eval()
    unet.eval()

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained(config['model_name'], subfolder='scheduler')
    scheduler.set_timesteps(config['num_inference_steps'])

    logger.info(f"\nGenerating images for {len(prompts)} prompts...\n")

    # Generate for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f'[{prompt_idx+1}/{len(prompts)}] Prompt: "{prompt}"')

        batch_prompts = [prompt] * num_images_per_prompt

        # Encode text with CLIP LoRA
        text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)
        text_features = clip_lora_trainer.model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features.unsqueeze(1).float()

        # Project to 768 dim
        encoder_hidden_states = text_projection(text_features)

        # Generate latents
        latents = torch.randn(
            (num_images_per_prompt, 4, config['image_size'] // 8, config['image_size'] // 8),
            device=device
        )
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for t in tqdm(scheduler.timesteps, desc='Generating', leave=False):
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        # Save
        for i, img in enumerate(images):
            img_path = output_dir / f'prompt_{prompt_idx+1}_sample_{i+1}.png'
            save_image(img, img_path)

        grid = make_grid(images, nrow=2, padding=2)
        grid_path = output_dir / f'prompt_{prompt_idx+1}_grid.png'
        save_image(grid, grid_path)

        logger.info(f'  → Saved to {grid_path}')

    # Save prompts
    with open(output_dir / 'prompts.txt', 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i+1}. {prompt}\n")

    logger.info(f'\nAll images saved to {output_dir}')


# ============================================================
# Main
# ============================================================

def custom_collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images,
        'caption': captions,
        'image_path': image_paths
    }


def main():
    """Main training function"""

    config = {
        # Model config
        'model_name': 'runwayml/stable-diffusion-v1-5',
        'lora_rank': 8,
        'lora_alpha': 16.0,

        # CLIP LoRA config
        'clip_lora_checkpoint': 'outputs/clip_lora/clip_lora_epoch_5.pt',
        'clip_lora_rank': 8,
        'clip_lora_alpha': 8.0,

        # Training config
        'data_dir': 'data/flickr30k',
        'batch_size': 4,  # 减小到 4 以节省显存
        'gradient_accumulation_steps': 4,  # 累积4步，有效 batch size = 16
        'num_epochs': 10,  # 减少到 10 个 epoch
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'use_mixed_precision': True,
        'max_train_samples': 16000,  # 只用 16k 样本快速测试

        # Image config
        'image_size': 512,

        # Sampling config
        'num_inference_steps': 20,  # 减少到 20 步（更快）
        'sample_freq': 2,  # 每 2 个 epoch 采样
        'checkpoint_freq': 2,  # 每 2 个 epoch 保存

        # Output config
        'save_dir': 'outputs/sd15_clip_lora',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'\n{"="*60}')
    logger.info(f'Stable Diffusion 1.5 + CLIP LoRA Training')
    logger.info(f'{"="*60}')
    logger.info(f'Device: {device}')
    logger.info(f'Model: {config["model_name"]}')
    logger.info(f'CLIP LoRA: {config["clip_lora_checkpoint"]}')
    logger.info(f'UNet LoRA Rank: {config["lora_rank"]}, Alpha: {config["lora_alpha"]}')
    logger.info(f'{"="*60}\n')

    # Load dataset
    logger.info(f'Loading dataset from {config["data_dir"]}...')
    try:
        train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            tokenizer=None,
            split='train',
            train_ratio=0.9
        )

        # Limit training samples for faster testing
        if 'max_train_samples' in config and config['max_train_samples'] > 0:
            original_size = len(train_dataset)
            train_dataset.captions = train_dataset.captions[:config['max_train_samples']]
            logger.info(f'Limited dataset from {original_size} to {len(train_dataset)} samples for faster training')

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )

        logger.info(f'Dataset size: {len(train_dataset)}')
        logger.info(f'Number of batches: {len(train_loader)}')
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    # Load CLIP LoRA
    logger.info(f'\nLoading CLIP LoRA from {config["clip_lora_checkpoint"]}...')
    try:
        clip_lora_trainer = CLIPLoRATrainer(
            model_name='ViT-B/32',
            rank=config['clip_lora_rank'],
            alpha=config['clip_lora_alpha'],
            device=device
        )
        clip_lora_trainer.load_checkpoint(config['clip_lora_checkpoint'])
        clip_lora_trainer.model.eval()
        # Freeze CLIP LoRA
        for param in clip_lora_trainer.model.parameters():
            param.requires_grad = False
        logger.info('✓ CLIP LoRA loaded and frozen')
    except Exception as e:
        logger.error(f'Failed to load CLIP LoRA: {e}')
        logger.info(f'Please make sure {config["clip_lora_checkpoint"]} exists')
        return

    # Load SD1.5 models
    logger.info('\nLoading Stable Diffusion 1.5 models...')
    try:
        dtype = torch.float16 if device.type == 'cuda' else torch.float32

        vae = AutoencoderKL.from_pretrained(
            config['model_name'],
            subfolder='vae',
            torch_dtype=dtype
        ).to(device)
        vae.enable_slicing()  # Enable VAE slicing for memory optimization
        logger.info('✓ VAE loaded (with slicing)')

        unet = UNet2DConditionModel.from_pretrained(
            config['model_name'],
            subfolder='unet',
            torch_dtype=dtype
        ).to(device)
        unet.enable_gradient_checkpointing()  # Enable gradient checkpointing
        logger.info('✓ UNet loaded (with gradient checkpointing)')

        noise_scheduler = DDPMScheduler.from_pretrained(
            config['model_name'],
            subfolder='scheduler'
        )
        logger.info('✓ Scheduler loaded')

    except Exception as e:
        logger.error(f'Failed to load models: {e}')
        return

    # Freeze UNet base parameters
    logger.info('\nFreezing UNet base parameters...')
    unet.requires_grad_(False)
    logger.info('✓ UNet frozen!')

    # Inject LoRA into UNet
    logger.info('\nInjecting LoRA into UNet...')
    inject_lora_into_unet(unet, rank=config['lora_rank'], alpha=config['lora_alpha'])

    # Create text projection layer (512 -> 768)
    logger.info('\nCreating text projection layer (512 -> 768)...')
    text_projection = TextProjection().to(device)
    logger.info('✓ Text projection created')

    # Train
    logger.info('\nStarting training...\n')
    train_sd15_with_clip_lora(
        vae=vae,
        clip_lora_trainer=clip_lora_trainer,
        text_projection=text_projection,
        unet=unet,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        config=config,
        device=device
    )

    logger.info('\n' + '='*60)
    logger.info('Training completed!')
    logger.info('='*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SD1.5 + CLIP LoRA Fine-tuning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                       help='Mode: train or generate')
    parser.add_argument('--clip-checkpoint', type=str,
                       default='outputs/clip_lora/clip_lora_epoch_5.pt',
                       help='CLIP LoRA checkpoint path')
    parser.add_argument('--unet-checkpoint', type=str,
                       default='outputs/sd15_clip_lora/final_model.pt',
                       help='UNet checkpoint path for generation')
    parser.add_argument('--prompt', type=str, action='append',
                       help='Text prompt for generation (can specify multiple times)')
    parser.add_argument('--num-images', type=int, default=4,
                       help='Number of images per prompt')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/sd15_clip_lora/generated',
                       help='Output directory for generated images')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'generate':
        config = {
            'model_name': 'runwayml/stable-diffusion-v1-5',
            'lora_rank': 8,
            'lora_alpha': 16.0,
            'clip_lora_rank': 8,
            'clip_lora_alpha': 8.0,
            'image_size': 512,
            'num_inference_steps': 30,
        }

        prompts = args.prompt or [
            'a beautiful mountain landscape with snow',
            'a sunset over the ocean with colorful clouds',
            'a dense forest with tall trees and sunlight',
            'people walking in a busy city street'
        ]

        generate_images_from_prompts(
            clip_lora_checkpoint=args.clip_checkpoint,
            unet_checkpoint=args.unet_checkpoint,
            prompts=prompts,
            config=config,
            output_dir=args.output_dir,
            num_images_per_prompt=args.num_images
        )

"""
Step 2 with CLIP LoRA: Text-to-Image Diffusion Model

基于 CLIP LoRA 的文本条件图像生成模型：
阶段1: train_clip_lora.py - 微调 CLIP 进行文图对齐
阶段2: 本文件 - 使用微调的 CLIP + 扩散模型进行文生图

优势：
- CLIP 强大的文本理解能力（相比 LSTM TextEncoder）
- LoRA 微调适配 Flickr30k 数据集
- 输出维度 512（相比原来的 256）

架构：
文本 → CLIP LoRA (冻结) → 文本嵌入 (512维) → UNet Diffusion → 生成图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import logging
from pathlib import Path

# Import from step2.py (reuse UNet architecture)
# CrossAttention 和 ConditionedBlock 在 TextConditionedUNet 内部使用，不需要直接导入
from step2 import (
    TextConditionedUNet,
    TextConditionedDDPM,
)

# Import from main.py (dataset)
from main import (
    Flickr30kDataset,
)

# Import CLIP LoRA trainer
from train_clip_lora import CLIPLoRATrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import clip
except ImportError:
    logger.error("CLIP not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    import clip


def train_clip_lora_ddpm(
    ddpm,
    clip_trainer,
    train_loader,
    config,
    num_epochs=100,
    learning_rate=2e-4,
    save_dir='outputs/clip_lora_ddpm',
    sample_freq=10,
    device='cuda',
    image_size=64,
    gradient_accumulation_steps=1,
    use_mixed_precision=False
):
    """
    Train text-conditioned DDPM with CLIP LoRA text encoder

    Args:
        ddpm: TextConditionedDDPM model
        clip_trainer: CLIPLoRATrainer instance (frozen, only for text encoding)
        train_loader: DataLoader for image-text pairs
        config: Configuration dict
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'samples').mkdir(exist_ok=True)

    # Freeze CLIP (只训练 UNet)
    for param in clip_trainer.model.parameters():
        param.requires_grad = False
    clip_trainer.model.eval()
    logger.info("CLIP LoRA frozen - only training UNet")

    # Optimizer - 只优化 UNet 参数
    optimizer = torch.optim.AdamW(ddpm.model.parameters(), lr=learning_rate, weight_decay=0.01)
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
        clip_trainer.model.eval()  # CLIP 始终保持 eval 模式

        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            captions = batch['caption']  # List of strings

            # Normalize images to [-1, 1]
            images = (images - 0.5) * 2.0

            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                # Encode text using CLIP LoRA (frozen)
                with torch.no_grad():
                    # Tokenize with CLIP tokenizer
                    text_tokens = clip.tokenize(captions, truncate=True).to(device)
                    # Get text embeddings from CLIP
                    text_features = clip_trainer.model.encode_text(text_tokens)
                    # Normalize (CLIP best practice)
                    text_features = F.normalize(text_features, dim=-1)
                    # Add sequence dimension: (batch_size, 512) -> (batch_size, 1, 512)
                    text_context = text_features.unsqueeze(1)

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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
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
            for idx in indices:
                sample = train_loader.dataset[idx.item()]
                captions.append(sample['caption'])

            # Encode text with CLIP
            with torch.no_grad():
                clip_trainer.model.eval()
                text_tokens = clip.tokenize(captions, truncate=True).to(device)
                text_features = clip_trainer.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)
                text_context = text_features.unsqueeze(1).float()  # Convert to float32 for DDPM

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

        # Save checkpoint every epoch
        checkpoint_path = save_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': ddpm.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'config': config,
            'clip_checkpoint': config['clip_checkpoint'],  # Reference to CLIP LoRA checkpoint
        }, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path}')

    final_path = save_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': ddpm.model.state_dict(),
        'config': config,
        'clip_checkpoint': config['clip_checkpoint'],
    }, final_path)
    logger.info(f'Final model saved to {final_path}')

    return ddpm


def generate_images(
    clip_checkpoint='outputs/clip_lora/clip_lora_epoch_1.pt',
    ddpm_checkpoint='outputs/clip_lora_ddpm/final_model.pt',
    prompts=None,
    num_samples=4,
    image_size=64,
    output_dir='outputs/clip_lora_ddpm/generated',
    device='cuda'
):
    """
    Generate images from text prompts using trained CLIP LoRA + DDPM

    Args:
        clip_checkpoint: Path to CLIP LoRA checkpoint
        ddpm_checkpoint: Path to DDPM checkpoint
        prompts: List of text prompts
        num_samples: Number of images to generate per prompt
        image_size: Size of generated images
        output_dir: Directory to save generated images
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Default prompts if not provided
    if prompts is None:
        prompts = [
            'a beautiful mountain landscape',
            'a sunset over the ocean',
            'a forest with tall trees',
            'people walking in a city street'
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load CLIP LoRA
    logger.info(f'Loading CLIP LoRA from {clip_checkpoint}...')
    clip_trainer = CLIPLoRATrainer(
        model_name='ViT-B/32',
        rank=8,
        alpha=8.0,
        device=device
    )
    clip_trainer.load_checkpoint(clip_checkpoint)
    clip_trainer.model.eval()

    # Load DDPM checkpoint
    logger.info(f'Loading DDPM from {ddpm_checkpoint}...')
    checkpoint = torch.load(ddpm_checkpoint, map_location=device)
    config = checkpoint['config']

    # Create DDPM model
    unet = TextConditionedUNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config['time_emb_dim'],
        text_emb_dim=512,  # CLIP embedding dimension
        base_channels=config['base_channels'],
        channel_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
    ).to(device)

    ddpm = TextConditionedDDPM(
        model=unet,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )

    ddpm.model.load_state_dict(checkpoint['model_state_dict'])
    ddpm.model.eval()

    logger.info('Generating images...')

    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f'\nPrompt {prompt_idx+1}/{len(prompts)}: "{prompt}"')

        # Repeat prompt for num_samples
        batch_prompts = [prompt] * num_samples

        # Encode text with CLIP
        with torch.no_grad():
            text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)
            text_features = clip_trainer.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            text_context = text_features.unsqueeze(1).float()  # Convert to float32 for DDPM

            # Generate images
            samples = ddpm.sample(text_context, channels=3, image_size=image_size)
            samples = (samples + 1.0) / 2.0
            samples = torch.clamp(samples, 0.0, 1.0)

        # Save individual images
        for i, img in enumerate(samples):
            img_path = output_dir / f'prompt_{prompt_idx+1}_sample_{i+1}.png'
            save_image(img, img_path)

        # Save grid
        grid = make_grid(samples, nrow=2, padding=2)
        grid_path = output_dir / f'prompt_{prompt_idx+1}_grid.png'
        save_image(grid, grid_path)
        logger.info(f'Saved to {grid_path}')

    # Save all prompts
    with open(output_dir / 'prompts.txt', 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f"{i+1}. {prompt}\n")

    logger.info(f'\nAll images saved to {output_dir}')


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles the case where 'tokens' is None (when not using custom tokenizer)
    """
    # Separate different fields
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    # Handle tokens - may be None if no tokenizer provided
    if batch[0]['tokens'] is not None:
        tokens = torch.stack([item['tokens'] for item in batch])
    else:
        tokens = None

    return {
        'image': images,
        'caption': captions,
        'tokens': tokens,
        'image_path': image_paths
    }


def main():
    """Main training function"""

    config = {
        # Data config
        'data_dir': 'data/flickr30k',
        'batch_size': 8,
        'gradient_accumulation_steps': 2,  # Effective batch size = 16
        'num_epochs': 100,
        'learning_rate': 2e-4,
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'image_size': 64,  # Start with 64x64
        'save_dir': 'outputs/clip_lora_ddpm',
        'sample_freq': 10,
        'use_mixed_precision': True,  # 使用混合精度加速

        # Model config
        'text_emb_dim': 512,  # CLIP output dimension (ViT-B/32)
        'time_emb_dim': 256,
        'base_channels': 96,
        'channel_mult': (1, 2, 2, 4),
        'num_res_blocks': 2,
        'use_attention': (False, True, True, False),

        # CLIP LoRA config
        'clip_checkpoint': 'outputs/clip_lora/clip_lora_epoch_3.pt',  # 从阶段3得到
        'clip_model': 'ViT-B/32',
        'lora_rank': 8,
        'lora_alpha': 8.0,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load dataset (不需要 tokenizer，因为用 CLIP 的)
    logger.info(f'Loading dataset from {config["data_dir"]}...')

    try:
        train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            tokenizer=None,  # 不需要自定义 tokenizer
            split='train',
            train_ratio=0.9
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True,
            collate_fn=custom_collate_fn  # 使用自定义 collate 函数处理 tokens=None
        )

        logger.info(f'Dataset size: {len(train_dataset)}')
        logger.info(f'Number of batches: {len(train_loader)}')

    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        logger.info('Please make sure you have the Flickr30k dataset in data/flickr30k directory')
        return

    # Load CLIP LoRA (frozen, only for encoding)
    logger.info('Loading CLIP LoRA text encoder...')
    clip_trainer = CLIPLoRATrainer(
        model_name=config['clip_model'],
        rank=config['lora_rank'],
        alpha=config['lora_alpha'],
        device=device
    )

    try:
        clip_trainer.load_checkpoint(config['clip_checkpoint'])
        logger.info(f'Loaded CLIP LoRA from {config["clip_checkpoint"]}')
    except Exception as e:
        logger.error(f'Failed to load CLIP checkpoint: {e}')
        logger.info('Please train CLIP LoRA first using: python train_clip_lora.py --mode train')
        return

    # Create UNet model
    logger.info('Creating text-conditioned UNet...')
    unet = TextConditionedUNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config['time_emb_dim'],
        text_emb_dim=config['text_emb_dim'],  # 512 for CLIP
        base_channels=config['base_channels'],
        channel_mult=config['channel_mult'],
        num_res_blocks=config['num_res_blocks'],
        use_attention=config['use_attention'],
    ).to(device)

    # Create DDPM
    logger.info('Creating text-conditioned DDPM...')
    ddpm = TextConditionedDDPM(
        model=unet,
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )

    # Train
    logger.info('Starting training...')
    train_clip_lora_ddpm(
        ddpm=ddpm,
        clip_trainer=clip_trainer,
        train_loader=train_loader,
        config=config,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        sample_freq=config['sample_freq'],
        device=device,
        image_size=config['image_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        use_mixed_precision=config['use_mixed_precision']
    )

    logger.info('Training completed!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CLIP LoRA Text-to-Image Diffusion')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                       help='Mode: train or generate')
    parser.add_argument('--clip-checkpoint', type=str, default='outputs/clip_lora/clip_lora_epoch_1.pt',
                       help='CLIP LoRA checkpoint path')
    parser.add_argument('--ddpm-checkpoint', type=str, default='outputs/clip_lora_ddpm/final_model.pt',
                       help='DDPM checkpoint path for generation')
    parser.add_argument('--prompt', type=str, action='append',
                       help='Text prompt for generation (can specify multiple times)')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples per prompt')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Generated image size')
    parser.add_argument('--output-dir', type=str, default='outputs/clip_lora_ddpm/generated',
                       help='Output directory for generated images')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'generate':
        generate_images(
            clip_checkpoint=args.clip_checkpoint,
            ddpm_checkpoint=args.ddpm_checkpoint,
            prompts=args.prompt,
            num_samples=args.num_samples,
            image_size=args.image_size,
            output_dir=args.output_dir
        )

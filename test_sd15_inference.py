"""
直接测试 Stable Diffusion 1.5 的文生图功能（不需要微调）

使用预训练的 runwayml/stable-diffusion-v1-5 模型直接生成图像
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_images(
    prompts,
    model_name='runwayml/stable-diffusion-v1-5',
    num_images_per_prompt=4,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    output_dir='outputs/sd15_test',
    device='cuda'
):
    """
    使用 Stable Diffusion 1.5 生成图像

    Args:
        prompts: 文本提示列表
        model_name: 模型名称
        num_images_per_prompt: 每个提示生成的图像数量
        num_inference_steps: 推理步数（越多越慢但质量更好）
        guidance_scale: 引导强度（7-9 较好）
        height: 图像高度
        width: 图像宽度
        output_dir: 输出目录
        device: 设备
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 加载 Stable Diffusion 组件（与 train_sd15_lora.py 相同的方式）
    logger.info(f'\nLoading Stable Diffusion 1.5 components...')
    logger.info(f'Model: {model_name}')

    try:
        # 确定数据类型
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        if dtype == torch.float16:
            logger.info('Using FP16 precision for faster inference')

        # 分别加载各个组件（这样可以使用本地缓存）
        logger.info('Loading VAE...')
        vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder='vae',
            torch_dtype=dtype,
            local_files_only=False  # 会先尝试本地缓存
        ).to(device)
        logger.info('✓ VAE loaded')

        logger.info('Loading Text Encoder...')
        text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder='text_encoder',
            torch_dtype=dtype,
            local_files_only=False
        ).to(device)
        logger.info('✓ Text Encoder loaded')

        logger.info('Loading Tokenizer...')
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            subfolder='tokenizer',
            local_files_only=False
        )
        logger.info('✓ Tokenizer loaded')

        logger.info('Loading UNet...')
        unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder='unet',
            torch_dtype=dtype,
            local_files_only=False
        ).to(device)
        logger.info('✓ UNet loaded')

        logger.info('Loading Scheduler...')
        scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder='scheduler',
            local_files_only=False
        )
        logger.info('✓ Scheduler loaded')

        # 手动构建 Pipeline
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,  # 禁用安全检查器
            feature_extractor=None,
            requires_safety_checker=False
        )

        # 启用内存优化
        if device.type == 'cuda':
            pipe.enable_attention_slicing()
            logger.info('✓ Attention slicing enabled (memory optimization)')

    except Exception as e:
        logger.error(f'Failed to load components: {e}')
        logger.error('Please check if the model is cached locally.')
        return

    logger.info('✓ Pipeline constructed successfully!\n')

    # 配置信息
    logger.info('='*60)
    logger.info('Generation Configuration:')
    logger.info(f'  Resolution: {width}x{height}')
    logger.info(f'  Inference steps: {num_inference_steps}')
    logger.info(f'  Guidance scale: {guidance_scale}')
    logger.info(f'  Images per prompt: {num_images_per_prompt}')
    logger.info('='*60 + '\n')

    # 生成图像
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f'[{prompt_idx+1}/{len(prompts)}] Generating: "{prompt}"')

        try:
            # 生成图像
            with torch.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                images = pipe(
                    prompt=prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                ).images

            # 保存图像
            for i, image in enumerate(images):
                img_path = output_dir / f'prompt_{prompt_idx+1}_sample_{i+1}.png'
                image.save(img_path)
                logger.info(f'  ✓ Saved: {img_path}')

        except Exception as e:
            logger.error(f'  ✗ Failed to generate for prompt "{prompt}": {e}')
            continue

    # 保存提示词
    prompts_path = output_dir / 'prompts.txt'
    with open(prompts_path, 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f'{i+1}. {prompt}\n')

    logger.info(f'\n{"="*60}')
    logger.info(f'All images saved to: {output_dir}')
    logger.info(f'{"="*60}')


def main():
    """Main function"""

    # 测试提示词（中文会被自动翻译）
    prompts = [
        'a beautiful mountain landscape with snow-capped peaks and a clear blue sky',
        'a serene sunset over the ocean with colorful clouds',
        'a cozy coffee shop interior with warm lighting and comfortable furniture',
        'a futuristic city skyline at night with neon lights',
        'a cute cat sitting on a windowsill looking outside',
        'a vibrant flower garden in full bloom during spring',
    ]

    logger.info('\n' + '='*60)
    logger.info('Stable Diffusion 1.5 - Direct Inference Test')
    logger.info('='*60)
    logger.info(f'Will generate images for {len(prompts)} prompts')
    logger.info('='*60 + '\n')

    generate_images(
        prompts=prompts,
        model_name='runwayml/stable-diffusion-v1-5',
        num_images_per_prompt=2,  # 每个提示生成2张图
        num_inference_steps=30,   # 30步足够（更快）
        guidance_scale=7.5,
        height=512,
        width=512,
        output_dir='outputs/sd15_test',
        device='cuda'
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Stable Diffusion 1.5 Inference')
    parser.add_argument('--prompt', type=str, action='append',
                       help='Custom text prompt (can specify multiple times)')
    parser.add_argument('--num-images', type=int, default=2,
                       help='Number of images per prompt')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of inference steps (20-50)')
    parser.add_argument('--guidance', type=float, default=7.5,
                       help='Guidance scale (7-9 recommended)')
    parser.add_argument('--size', type=int, default=512,
                       help='Image size (512 or 768)')
    parser.add_argument('--output-dir', type=str, default='outputs/sd15_test',
                       help='Output directory')

    args = parser.parse_args()

    if args.prompt:
        # 使用用户提供的提示词
        prompts = args.prompt
    else:
        # 使用默认提示词
        prompts = [
            'a beautiful mountain landscape with snow-capped peaks and a clear blue sky',
            'a serene sunset over the ocean with colorful clouds',
            'a cozy coffee shop interior with warm lighting',
            'a futuristic city skyline at night with neon lights',
        ]

    logger.info('\n' + '='*60)
    logger.info('Stable Diffusion 1.5 - Direct Inference Test')
    logger.info('='*60)
    logger.info(f'Will generate images for {len(prompts)} prompts')
    logger.info('='*60 + '\n')

    generate_images(
        prompts=prompts,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.size,
        width=args.size,
        output_dir=args.output_dir,
    )

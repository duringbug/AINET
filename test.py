"""
Test script for text-to-image generation
"""

import torch
from torchvision.utils import save_image, make_grid
from pathlib import Path
import argparse
import logging

# Import from step2.py and main.py
from step2 import TextConditionedUNet, TextConditionedDDPM
from main import TextEncoder, SimpleTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir='outputs/text_conditioned_ddpm/checkpoints'):
    """Find the latest checkpoint automatically"""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

    if not checkpoints:
        return None

    # Sort by modification time, return the latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


class TextToImageTester:
    """Test text-to-image generation"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize tester with trained model

        Args:
            checkpoint_path: Path to checkpoint from step2.py training
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config from checkpoint if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            # Default config (should match training config)
            self.config = {
                'text_emb_dim': 256,
                'time_emb_dim': 256,
                'base_channels': 64,
                'channel_mult': (1, 2, 2),
                'num_res_blocks': 1,
                'use_attention': (False, True, True),
                'num_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'image_size': 64,
            }

        # Initialize tokenizer
        if 'tokenizer' in checkpoint:
            logger.info("Loading tokenizer from checkpoint...")
            self.tokenizer = SimpleTokenizer(
                vocab_size=checkpoint['tokenizer']['vocab_size'],
                max_length=checkpoint['tokenizer']['max_length']
            )
            self.tokenizer.word2idx = checkpoint['tokenizer']['word2idx']
            self.tokenizer.idx2word = checkpoint['tokenizer']['idx2word']
        else:
            # Try to load from pretrained checkpoint
            logger.warning("Tokenizer not found in checkpoint, attempting to load from pretrained model...")
            pretrained = torch.load('outputs/best_model/pytorch_model.bin', map_location=self.device)
            self.tokenizer = SimpleTokenizer(
                vocab_size=pretrained['tokenizer']['vocab_size'],
                max_length=pretrained['tokenizer']['max_length']
            )
            self.tokenizer.word2idx = pretrained['tokenizer']['word2idx']
            self.tokenizer.idx2word = pretrained['tokenizer']['idx2word']

        # Initialize text encoder
        logger.info("Initializing text encoder...")
        self.text_encoder = TextEncoder(
            vocab_size=len(self.tokenizer.word2idx),
            embed_dim=self.config.get('text_emb_dim', 256)
        ).to(self.device)

        # Load text encoder weights
        if 'text_encoder_state_dict' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
            logger.info("Text encoder loaded from checkpoint")
        else:
            logger.warning("Text encoder not in checkpoint, loading from pretrained...")
            pretrained = torch.load('outputs/best_model/pytorch_model.bin', map_location=self.device)
            text_encoder_state = {}
            for key, value in pretrained['model_state_dict'].items():
                if key.startswith('text_encoder.'):
                    new_key = key.replace('text_encoder.', '')
                    text_encoder_state[new_key] = value
            self.text_encoder.load_state_dict(text_encoder_state, strict=False)

        self.text_encoder.eval()

        # Initialize UNet
        logger.info("Initializing UNet...")
        self.unet = TextConditionedUNet(
            in_channels=3,
            out_channels=3,
            time_emb_dim=self.config.get('time_emb_dim', 256),
            text_emb_dim=self.config.get('text_emb_dim', 256),
            base_channels=self.config.get('base_channels', 64),
            channel_mult=self.config.get('channel_mult', (1, 2, 2)),
            num_res_blocks=self.config.get('num_res_blocks', 1),
            use_attention=self.config.get('use_attention', (False, True, True)),
        ).to(self.device)

        # Load UNet weights
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.unet.eval()

        # Initialize DDPM
        logger.info("Initializing DDPM...")
        self.ddpm = TextConditionedDDPM(
            model=self.unet,
            num_timesteps=self.config.get('num_timesteps', 1000),
            beta_start=self.config.get('beta_start', 0.0001),
            beta_end=self.config.get('beta_end', 0.02),
            device=self.device
        )

        logger.info("Model loaded successfully!")
        total_params = sum(p.numel() for p in self.unet.parameters())
        logger.info(f"UNet parameters: {total_params:,}")

    @torch.no_grad()
    def generate(self, texts, num_samples_per_text=1):
        """
        Generate images from text descriptions

        Args:
            texts: List of text descriptions or single text string
            num_samples_per_text: Number of images to generate per text

        Returns:
            Generated images as tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        all_images = []
        all_captions = []

        for text in texts:
            logger.info(f"Generating {num_samples_per_text} image(s) for: '{text}'")

            # Tokenize and encode text
            tokens = self.tokenizer.encode(text)
            tokens = torch.tensor([tokens] * num_samples_per_text, dtype=torch.long).to(self.device)

            # Encode text to embeddings
            text_embeddings = self.text_encoder(tokens)  # (num_samples, text_emb_dim)
            text_context = text_embeddings.unsqueeze(1)  # (num_samples, 1, text_emb_dim)

            # Generate images
            generated = self.ddpm.sample(
                text_context,
                channels=3,
                image_size=self.config.get('image_size', 64)
            )

            # Convert from [-1, 1] to [0, 1]
            generated = (generated + 1.0) / 2.0
            generated = torch.clamp(generated, 0.0, 1.0)

            all_images.append(generated)
            all_captions.extend([text] * num_samples_per_text)

        # Concatenate all generated images
        all_images = torch.cat(all_images, dim=0)

        return all_images, all_captions

    def save_images(self, images, captions, output_dir, prefix='generated'):
        """
        Save generated images

        Args:
            images: Tensor of images (N, C, H, W)
            captions: List of captions
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save individual images
        for i, (img, caption) in enumerate(zip(images, captions)):
            # Save image
            img_path = output_dir / f'{prefix}_{i+1}.png'
            save_image(img, img_path)
            logger.info(f"Saved: {img_path}")

        # Save grid
        grid = make_grid(images, nrow=min(4, len(images)), padding=2)
        grid_path = output_dir / f'{prefix}_grid.png'
        save_image(grid, grid_path)
        logger.info(f"Saved grid: {grid_path}")

        # Save captions
        caption_path = output_dir / f'{prefix}_captions.txt'
        with open(caption_path, 'w') as f:
            for i, caption in enumerate(captions):
                f.write(f"{i+1}. {caption}\n")
        logger.info(f"Saved captions: {caption_path}")

        return grid_path


def main():
    parser = argparse.ArgumentParser(description='Test text-to-image generation')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint (default: auto-find latest)')
    parser.add_argument('--text', type=str, nargs='+',
                       help='Text descriptions (can provide multiple)')
    parser.add_argument('--text-file', type=str,
                       help='File containing text descriptions (one per line)')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of images to generate per text')
    parser.add_argument('--output-dir', type=str, default='outputs/test_results',
                       help='Output directory for generated images')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--prefix', type=str, default='generated',
                       help='Prefix for output filenames')

    args = parser.parse_args()

    # Collect texts
    texts = []

    if args.text:
        texts.extend(args.text)

    if args.text_file:
        with open(args.text_file, 'r') as f:
            texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        # Default example texts
        logger.info("No text provided, using default examples...")
        texts = [
            "a dog running on the beach",
            "a beautiful sunset over mountains",
            "a cat sitting on a couch",
            "a person riding a bicycle in the park",
            "flowers in a garden",
            "a city street at night",
            "a bird flying in the sky",
            "children playing in the park"
        ]

    logger.info(f"Generating images for {len(texts)} text(s)...")

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        logger.info("No checkpoint specified, searching for latest...")
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint found! Please train the model first or specify --checkpoint")
            return
        logger.info(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return

    # Initialize tester
    tester = TextToImageTester(
        checkpoint_path=str(checkpoint_path),
        device=args.device
    )

    # Generate images
    images, captions = tester.generate(texts, num_samples_per_text=args.num_samples)

    logger.info(f"Generated {len(images)} images")

    # Save images
    tester.save_images(
        images,
        captions,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

    logger.info("Done!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

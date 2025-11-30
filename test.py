"""
Text-to-Image Generation Testing Script

This script loads a trained GenerativeMultimodalModel and generates images from text descriptions.
"""

import torch
from pathlib import Path
from PIL import Image
import json
import logging
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Import model class from main.py
from main import GenerativeMultimodalModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextToImageGenerator:
    """Text-to-Image generator using trained multimodal model"""

    def __init__(self, checkpoint_path, device=None):
        """
        Args:
            checkpoint_path: Path to model checkpoint directory
            device: Device to run on (cuda/cpu/mps), auto-detect if None
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Setup device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info("Using CUDA GPU")
            else:
                self.device = torch.device('cpu')
                logger.info("Using CPU")
        else:
            self.device = torch.device(device)
            logger.info(f"Using device: {device}")

        # Load config
        config_path = self.checkpoint_path.parent / 'config.json'
        if not config_path.exists():
            # Try in checkpoint_path itself
            config_path = self.checkpoint_path / 'config.json'
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        logger.info(f"Loaded config from {config_path}")

        # Load tokenizer
        tokenizer_path = self.checkpoint_path / 'tokenizer'
        if tokenizer_path.exists():
            self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            # Fallback to default BERT tokenizer
            bert_model_name = self.config.get('bert_model_name', 'bert-base-uncased')
            cache_dir = self.config.get('cache_dir', './models/bert_cache')
            self.tokenizer = BertTokenizer.from_pretrained(
                bert_model_name,
                cache_dir=cache_dir
            )
            logger.info(f"Loaded default tokenizer: {bert_model_name}")

        # Initialize model
        self.model = GenerativeMultimodalModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.config.get('embed_dim', 512),
            use_simple_cnn=self.config.get('use_simple_cnn', True),
            image_size=self.config.get('image_size', 224),
            max_text_length=self.config.get('max_text_length', 77),
            num_diffusion_steps=self.config.get('num_diffusion_steps', 1000),
            bert_model_name=self.config.get('bert_model_name', 'bert-base-uncased'),
            freeze_bert=False,  # Need gradients for generation
            cache_dir=self.config.get('cache_dir', './models/bert_cache')
        ).to(self.device)

        # Load checkpoint
        checkpoint_file = self.checkpoint_path / 'pytorch_model.bin'
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model checkpoint from {checkpoint_file}")
        logger.info(f"Model trained for {checkpoint.get('global_step', 'unknown')} steps")

    @torch.no_grad()
    def generate_from_text(self, text_prompts, num_samples_per_prompt=1, use_diffusion=True,
                          num_inference_steps=50, seed=None):
        """
        Generate images from text descriptions using diffusion model

        Args:
            text_prompts: List of text descriptions or single string
            num_samples_per_prompt: Number of images to generate per prompt
            use_diffusion: If True, use multi-step diffusion (slow but better);
                          If False, use direct decoder (fast but lower quality)
            num_inference_steps: Number of diffusion steps (50-1000, higher = slower but better)
            seed: Random seed for reproducibility (None = random every time)

        Returns:
            List of generated images (PIL Images)
        """
        # Handle single string input
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        all_images = []

        for prompt in text_prompts:
            if use_diffusion:
                logger.info(f"Generating image for: '{prompt}' (diffusion steps={num_inference_steps})")
            else:
                logger.info(f"Generating image for: '{prompt}' (direct decoding)")

            # Tokenize text
            encoding = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                max_length=self.config.get('max_text_length', 77),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Generate multiple samples with different random noise
            for i in range(num_samples_per_prompt):
                # Set seed for this specific sample if provided
                if seed is not None:
                    torch.manual_seed(seed + i)

                # Encode text to embedding space
                text_embedding = self.model.text_encoder(input_ids, attention_mask)

                if use_diffusion:
                    # Use diffusion model for multi-step iterative generation
                    # This produces higher quality but is slower
                    generated_embedding = self.model.diffusion.sample(
                        batch_size=1,
                        device=self.device,
                        condition=text_embedding,
                        num_inference_steps=num_inference_steps
                    )
                else:
                    # Direct decoding without diffusion (faster but lower quality)
                    generated_embedding = text_embedding

                # Decode embedding to image
                generated_image = self.model.image_decoder(generated_embedding)

                # Convert from [-1, 1] to [0, 1]
                generated_image = (generated_image + 1.0) / 2.0

                # Clamp to valid range
                generated_image = torch.clamp(generated_image, 0, 1)

                # Convert to PIL Image
                image_np = generated_image[0].cpu().permute(1, 2, 0).numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                all_images.append({
                    'prompt': prompt,
                    'image': image_pil,
                    'sample_id': i,
                    'use_diffusion': use_diffusion,
                    'num_steps': num_inference_steps if use_diffusion else 1
                })

        logger.info(f"Generated {len(all_images)} images")
        return all_images

    @torch.no_grad()
    def generate_from_diffusion(self, num_samples=4):
        """
        Generate images using the diffusion model (unconditional generation)

        Args:
            num_samples: Number of images to generate

        Returns:
            List of generated images (PIL Images)
        """
        logger.info(f"Generating {num_samples} images from diffusion model...")

        # Generate embeddings from diffusion
        generated_embeddings = self.model.diffusion.sample(num_samples, self.device)

        # Decode to images
        generated_images = self.model.image_decoder(generated_embeddings)

        # Convert from [-1, 1] to [0, 1]
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0, 1)

        # Convert to PIL Images
        all_images = []
        for i in range(num_samples):
            image_np = generated_images[i].cpu().permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            all_images.append({
                'prompt': 'diffusion_generated',
                'image': image_pil,
                'sample_id': i
            })

        logger.info(f"Generated {len(all_images)} images from diffusion")
        return all_images


def save_images(results, output_dir='test_outputs', grid=True):
    """
    Save generated images

    Args:
        results: List of result dicts from generate_from_text
        output_dir: Directory to save images
        grid: Whether to create a grid visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save individual images
    for idx, result in enumerate(results):
        prompt = result['prompt']
        image = result['image']
        sample_id = result['sample_id']

        # Create safe filename from prompt
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50]  # Limit length

        filename = f"{idx:03d}_{safe_prompt}_sample{sample_id}.png"
        filepath = output_path / filename

        image.save(filepath)
        logger.info(f"Saved: {filepath}")

    # Create grid visualization
    if grid and len(results) > 1:
        n_images = len(results)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, result in enumerate(results):
            row = idx // n_cols
            col = idx % n_cols

            axes[row][col].imshow(result['image'])
            axes[row][col].set_title(result['prompt'][:40], fontsize=10)
            axes[row][col].axis('off')

        # Hide empty subplots
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis('off')

        plt.tight_layout()
        grid_path = output_path / 'grid.png'
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved grid visualization: {grid_path}")
        plt.close(fig)  # Close the specific figure to free memory


def main():
    """Main testing function"""

    # Find best model
    outputs_dir = Path('outputs')
    best_model_path = outputs_dir / 'best_model'

    if not best_model_path.exists():
        # Fallback to latest epoch
        epoch_dirs = sorted(outputs_dir.glob('epoch-*'), key=lambda p: int(p.name.split('-')[1]))
        if epoch_dirs:
            best_model_path = epoch_dirs[-1]
            logger.warning(f"best_model not found, using {best_model_path}")
        else:
            raise FileNotFoundError("No trained model found in outputs/")

    logger.info(f"Using model from: {best_model_path}")

    # Initialize generator
    generator = TextToImageGenerator(best_model_path)

    # Test prompts
    test_prompts = [
        "a dog playing in the park",
        "a beautiful sunset over the ocean",
        "a red car on the street",
        "children playing soccer",
        "a cat sitting on a chair",
        "people walking on the beach",
    ]

    logger.info(f"\n{'='*60}")
    logger.info("Testing Text-to-Image Generation with Diffusion Model")
    logger.info(f"{'='*60}\n")

    # Generate images using multi-step diffusion process
    # Each iteration gradually refines the image (like Stable Diffusion)
    results = generator.generate_from_text(
        test_prompts,
        num_samples_per_prompt=2,  # Generate 2 different versions per prompt
        use_diffusion=True,  # Use multi-step iterative diffusion
        num_inference_steps=50,  # 50 denoising steps (balance between speed and quality)
        seed=None  # Random seed for variety
    )

    # Save results
    save_images(results, output_dir='test_outputs', grid=True)

    logger.info(f"\n{'='*60}")
    logger.info("Testing completed! Check test_outputs/ directory")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()

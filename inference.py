"""
Inference script for image-to-text and text-to-image generation
"""

import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer
import logging
import argparse
import matplotlib.pyplot as plt

# Import model from main.py
from main import GenerativeMultimodalModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageTextGenerator:
    """Wrapper class for image-text generation"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to checkpoint directory (e.g., 'outputs/best_model')
            device: 'cuda', 'cpu', or 'mps'
        """
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        checkpoint_path = Path(checkpoint_path)
        model_file = checkpoint_path / 'pytorch_model.bin'

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(model_file, map_location=self.device)
        config = checkpoint['config']

        # Initialize BERT tokenizer
        logger.info("Loading BERT tokenizer...")
        bert_model_name = config.get('bert_model_name', 'bert-base-uncased')
        cache_dir = config.get('cache_dir', './models/bert_cache')

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        self.tokenizer.model_max_length = config.get('max_text_length', 77)

        # Initialize model
        logger.info("Initializing model...")
        self.model = GenerativeMultimodalModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=config.get('embed_dim', 512),
            use_simple_cnn=config.get('use_simple_cnn', True),
            image_size=config.get('image_size', 224),
            max_text_length=config.get('max_text_length', 77),
            num_diffusion_steps=config.get('num_diffusion_steps', 1000),
            bert_model_name=bert_model_name,
            freeze_bert=False,  # Always unfreeze for inference
            cache_dir=cache_dir
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info("Model loaded successfully!")

        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def image_to_text(self, image_path, return_attention=False):
        """
        Generate text caption from image

        Args:
            image_path: Path to input image
            return_attention: Whether to return attention weights (not implemented yet)

        Returns:
            caption: Generated text caption
        """
        logger.info(f"Generating caption for: {image_path}")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # Encode image
        image_embeddings = self.model.image_encoder(image_tensor)

        # Generate text
        tokens = self.model.generate_text_from_image(image_embeddings)

        # Debug: Print raw tokens
        logger.info(f"Raw tokens: {tokens[0].tolist()[:20]}")  # First 20 tokens

        # Decode tokens to text
        caption = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

        # If empty, try without filtering special tokens
        if not caption.strip():
            caption_with_special = self.tokenizer.decode(tokens[0], skip_special_tokens=False)
            logger.warning(f"Caption is empty! Raw decode: {caption_with_special[:100]}")

        logger.info(f"Generated caption: {caption}")
        return caption

    @torch.no_grad()
    def text_to_image(self, text, num_inference_steps=50, save_path=None):
        """
        Generate image from text description

        Args:
            text: Text description
            num_inference_steps: Number of diffusion steps (10-100)
                - More steps = better quality but slower
                - Recommended: 20-50 for good results
            save_path: Optional path to save generated image

        Returns:
            image: PIL Image
        """
        logger.info(f"Generating image from text: {text}")
        logger.info(f"Using {num_inference_steps} diffusion steps")

        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Encode text
        text_embeddings = self.model.text_encoder(input_ids, attention_mask)

        # Generate image via latent diffusion
        generated_image_tensor = self.model.generate_image_from_text(
            text_embeddings,
            num_inference_steps=num_inference_steps
        )

        # Convert to PIL Image
        image_tensor = generated_image_tensor[0].cpu()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray((image_np * 255).astype('uint8'))

        if save_path:
            image.save(save_path)
            logger.info(f"Image saved to: {save_path}")

        return image

    @torch.no_grad()
    def batch_text_to_image(self, texts, num_inference_steps=50, save_dir=None):
        """
        Generate multiple images from text descriptions

        Args:
            texts: List of text descriptions
            num_inference_steps: Number of diffusion steps
            save_dir: Optional directory to save images

        Returns:
            images: List of PIL Images
        """
        images = []

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        for i, text in enumerate(texts):
            save_path = save_dir / f"generated_{i}.png" if save_dir else None
            image = self.text_to_image(text, num_inference_steps, save_path)
            images.append(image)

        return images

    @torch.no_grad()
    def batch_image_to_text(self, image_paths):
        """
        Generate captions for multiple images

        Args:
            image_paths: List of image paths

        Returns:
            captions: List of generated captions
        """
        captions = []

        for image_path in image_paths:
            caption = self.image_to_text(image_path)
            captions.append(caption)

        return captions


def main():
    parser = argparse.ArgumentParser(description='Image-Text Generation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory (e.g., outputs/best_model)')
    parser.add_argument('--mode', type=str, required=True, choices=['i2t', 't2i', 'both'],
                       help='Generation mode: i2t (image-to-text), t2i (text-to-image), both')
    parser.add_argument('--image', type=str,
                       help='Input image path (for i2t mode)')
    parser.add_argument('--text', type=str,
                       help='Input text (for t2i mode)')
    parser.add_argument('--output', type=str, default='generated.png',
                       help='Output image path (for t2i mode)')
    parser.add_argument('--steps', type=int, default=50,
                       help='Number of diffusion steps for t2i (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')

    args = parser.parse_args()

    # Initialize generator
    generator = ImageTextGenerator(args.checkpoint, device=args.device)

    # Image-to-Text
    if args.mode in ['i2t', 'both']:
        if not args.image:
            logger.error("--image is required for i2t mode")
            return

        caption = generator.image_to_text(args.image)
        print("\n" + "="*60)
        print(f"Image: {args.image}")
        print(f"Generated Caption: {caption}")
        print("="*60 + "\n")

    # Text-to-Image
    if args.mode in ['t2i', 'both']:
        if not args.text:
            logger.error("--text is required for t2i mode")
            return

        image = generator.text_to_image(args.text, num_inference_steps=args.steps, save_path=args.output)
        print("\n" + "="*60)
        print(f"Text: {args.text}")
        print(f"Generated Image: {args.output}")
        print("="*60 + "\n")

        # Display image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Generated: "{args.text}"')
        plt.tight_layout()
        plt.savefig(args.output.replace('.png', '_preview.png'), dpi=150, bbox_inches='tight')
        logger.info(f"Preview saved to: {args.output.replace('.png', '_preview.png')}")


if __name__ == '__main__':
    main()

"""
Image-to-Text Generation Testing Script

This script loads a trained GenerativeMultimodalModel and generates text descriptions from images.
"""

import torch
from pathlib import Path
from PIL import Image
import json
import logging
from transformers import BertTokenizer
import torchvision.transforms as transforms

# Import model class from main.py
from main import GenerativeMultimodalModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageToTextGenerator:
    """Image-to-Text generator using trained multimodal model"""

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
            freeze_bert=False,
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

        # Image preprocessing (same as training)
        image_size = self.config.get('image_size', 224)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def generate_from_image(self, image_paths, num_captions_per_image=3,
                           seed=None, temperature=1.0,
                           top_k=50, top_p=0.9, repetition_penalty=1.2):
        """
        Generate text descriptions from images using autoregressive decoding

        Image-to-Text generation uses Transformer autoregressive decoding,
        NOT diffusion (diffusion is only for Text-to-Image).

        Args:
            image_paths: List of image paths or single path
            num_captions_per_image: Number of different captions to generate per image
            seed: Random seed for reproducibility (None = random)
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
                - 0.7-0.8: More focused, coherent captions
                - 0.9-1.0: More diverse captions
            top_k: Keep only top k tokens for sampling (0 = disabled)
            top_p: Nucleus sampling probability (0-1, closer to 1 = more diverse)
            repetition_penalty: Penalty for repeating tokens (>1 discourages repetition)

        Returns:
            List of result dicts with 'image_path', 'caption', 'tokens'
        """
        # Handle single path input
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        all_results = []

        for img_path in image_paths:
            img_path = Path(img_path)

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            logger.info(f"Processing image: {img_path}")

            # Load and preprocess image
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

            # Generate multiple captions with different random variations
            for i in range(num_captions_per_image):
                # Set seed if provided
                if seed is not None:
                    torch.manual_seed(seed + i)

                logger.info(f"  Generating caption {i+1}/{num_captions_per_image}")

                # Encode image to embedding space
                image_embedding = self.model.image_encoder(image_tensor)

                # Generate tokens using autoregressive decoding with sampling
                # No diffusion needed - Transformer handles text generation well
                predicted_tokens = self.generate_tokens_with_sampling(
                    image_embedding,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )

                # Convert tokens to text
                caption = self.decode_tokens(predicted_tokens)

                all_results.append({
                    'image_path': str(img_path),
                    'caption': caption,
                    'tokens': predicted_tokens.cpu().tolist(),
                    'sample_id': i
                })

                logger.info(f"    Caption: {caption}")

        logger.info(f"Generated {len(all_results)} captions for {len(image_paths)} images")
        return all_results

    @torch.no_grad()
    def generate_tokens_with_sampling(self, embedding, max_length=None, temperature=1.0,
                                     top_k=50, top_p=0.9, repetition_penalty=1.2):
        """
        Generate tokens autoregressively with Transformer decoder

        Args:
            embedding: Input embedding (1, embed_dim)
            max_length: Maximum sequence length (None = use model default)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Tensor of generated tokens (max_length,)
        """
        if max_length is None:
            max_length = self.config.get('max_text_length', 77)

        device = embedding.device

        # Project condition embedding
        condition = self.model.text_decoder.condition_proj(embedding).unsqueeze(1)

        # Start with [CLS] token
        generated = torch.tensor([[101]], dtype=torch.long, device=device)
        generated_tokens = []
        token_counts = {}

        for step in range(max_length):
            # Embed current sequence
            x = self.model.text_decoder.word_embedding(generated)
            seq_len = x.size(1)
            # Add positional encoding
            x = x + self.model.text_decoder.pos_encoding[:, :seq_len, :]
            x = self.model.text_decoder.dropout(x)

            # Generate causal mask
            tgt_mask = self.model.text_decoder.generate_causal_mask(seq_len, device)

            # Pass through transformer layers
            for layer in self.model.text_decoder.layers:
                x = layer(x, condition, tgt_mask)

            # Get logits for last position
            logits = self.model.text_decoder.output_projection(x[:, -1, :])  # (1, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token_id in set(generated_tokens):
                    if token_id in token_counts:
                        logits[0, token_id] /= (repetition_penalty ** token_counts[token_id])

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Get token ID
            token_id = next_token[0, 0].item()

            # Stop if we hit [SEP] (102) or [PAD] (0)
            if token_id in [0, 102]:
                break

            # Add to generated sequence
            generated_tokens.append(token_id)
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # Append to sequence for next iteration
            generated = torch.cat([generated, next_token], dim=1)

        # Convert to tensor
        if len(generated_tokens) == 0:
            generated_tokens = [0]

        result = torch.tensor(generated_tokens, dtype=torch.long, device=device)
        return result

    def decode_tokens(self, tokens):
        """
        Convert token IDs to readable text

        Args:
            tokens: Tensor of token IDs (seq_len,)

        Returns:
            Decoded text string
        """
        # Convert to list if tensor
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        # Decode using tokenizer
        caption = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Clean up the text
        caption = caption.strip()

        # Remove excessive spaces
        caption = ' '.join(caption.split())

        return caption


def save_results(results, output_file='test02_results.txt'):
    """
    Save generated captions to a text file

    Args:
        results: List of result dicts from generate_from_image
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Image-to-Text Generation Results\n")
        f.write("=" * 80 + "\n\n")

        current_image = None
        for result in results:
            # Write image path header when switching to new image
            if result['image_path'] != current_image:
                current_image = result['image_path']
                f.write(f"\nImage: {current_image}\n")
                f.write("-" * 80 + "\n")

            # Write caption
            sample_id = result['sample_id']
            caption = result['caption']

            f.write(f"Caption {sample_id + 1}: {caption}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Total: {len(results)} captions generated\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Results saved to: {output_path}")


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
    generator = ImageToTextGenerator(best_model_path)

    # Find all images in test/picture directory
    test_image_dir = Path('test/picture')

    if not test_image_dir.exists():
        logger.error(f"Test image directory not found: {test_image_dir}")
        return

    # Get all image files (jpg, png, jpeg)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(test_image_dir.glob(ext))

    if not image_files:
        logger.error(f"No images found in {test_image_dir}")
        return

    image_files = sorted(image_files)
    logger.info(f"Found {len(image_files)} images in {test_image_dir}")

    logger.info(f"\n{'='*80}")
    logger.info("Testing Image-to-Text Generation")
    logger.info(f"{'='*80}\n")

    # Generate captions for all images
    # Try different decoding strategies for better quality
    all_results = []

    # Method 1: Sampling with diversity (more random, creative captions)
    logger.info("\n--- Method 1: Diverse Sampling (Higher Temperature) ---\n")
    results_sampling = generator.generate_from_image(
        image_files,
        num_captions_per_image=3,  # Generate 3 different captions
        temperature=0.9,  # Higher temperature for diversity
        top_k=50,  # Top-k sampling
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.3,  # Strong penalty against repetition
        seed=None  # Random for diverse results
    )
    all_results.extend(results_sampling)

    # Method 2: Focused sampling (more deterministic, coherent captions)
    logger.info("\n--- Method 2: Focused Sampling (Lower Temperature) ---\n")
    results_focused = generator.generate_from_image(
        image_files,
        num_captions_per_image=2,  # Generate 2 captions
        temperature=0.7,  # Lower temperature for more focused results
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.5,  # Even stronger penalty
        seed=42  # Fixed seed for reproducibility
    )
    all_results.extend(results_focused)

    # Save all results
    save_results(all_results, output_file='test02_results.txt')

    logger.info(f"\n{'='*80}")
    logger.info("Testing completed!")
    logger.info(f"Generated {len(all_results)} captions for {len(image_files)} images")
    logger.info("Results saved to: test02_results.txt")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()

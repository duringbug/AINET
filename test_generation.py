"""
Quick test script for image-text generation
"""

import torch
from pathlib import Path
from inference import ImageTextGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_generation():
    """Test both image-to-text and text-to-image generation"""

    # Find latest checkpoint
    output_dir = Path('outputs')
    checkpoint_path = output_dir / 'best_model'

    if not checkpoint_path.exists():
        # Try to find any checkpoint
        checkpoints = sorted(output_dir.glob('epoch-*'))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            logger.error("No checkpoint found in outputs/")
            logger.info("Please train the model first using: python main.py")
            return

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Initialize generator
    generator = ImageTextGenerator(checkpoint_path, device=device)

    # Test 1: Text-to-Image Generation
    print("\n" + "="*80)
    print("TEST 1: Text-to-Image Generation")
    print("="*80)

    test_prompts = [
        "a dog playing in the park",
        "a cat sitting on a table",
        "a beautiful sunset over the ocean",
        "a person riding a bicycle",
        "a red car on the street"
    ]

    output_dir = Path('generated_images')
    output_dir.mkdir(exist_ok=True)

    for i, prompt in enumerate(test_prompts):
        print(f"\nGenerating image {i+1}/{len(test_prompts)}: '{prompt}'")
        save_path = output_dir / f"text2img_{i+1}.png"

        image = generator.text_to_image(
            prompt,
            num_inference_steps=30,  # Use 30 steps for faster generation
            save_path=save_path
        )

        print(f"✓ Saved to: {save_path}")

    # Test 2: Image-to-Text Generation
    print("\n" + "="*80)
    print("TEST 2: Image-to-Text Generation")
    print("="*80)

    # Try to find some test images from the dataset
    data_dir = Path('data/coco/coco2017')

    if data_dir.exists():
        # Find validation images
        val_images = data_dir / 'val2017'
        if val_images.exists():
            test_images = sorted(val_images.glob('*.jpg'))[:5]

            if test_images:
                for i, img_path in enumerate(test_images):
                    print(f"\nGenerating caption {i+1}/{len(test_images)}: {img_path.name}")
                    caption = generator.image_to_text(img_path)
                    print(f"✓ Caption: '{caption}'")
            else:
                logger.warning("No validation images found")
        else:
            logger.warning(f"Validation directory not found: {val_images}")
    else:
        logger.warning(f"Dataset directory not found: {data_dir}")
        logger.info("To test image-to-text, please provide image paths manually")

    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80)
    print(f"\nGenerated images saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    test_generation()

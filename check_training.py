"""Check training progress"""
import torch
from pathlib import Path
import json

# Find latest checkpoint
output_dir = Path('outputs')
checkpoints = sorted(output_dir.glob('checkpoint-step-*'),
                     key=lambda x: int(x.name.split('-')[-1]))

if checkpoints:
    latest_ckpt = checkpoints[-1]
    print(f"Latest checkpoint: {latest_ckpt.name}")

    # Load checkpoint
    ckpt_file = latest_ckpt / 'pytorch_model.bin'
    if ckpt_file.exists():
        checkpoint = torch.load(ckpt_file, map_location='cpu')

        print(f"\nTraining Progress:")
        print(f"  Global Step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Best Loss: {checkpoint.get('best_loss', 'N/A'):.4f}")

        config = checkpoint.get('config', {})
        batch_size = config.get('batch_size', 64)
        num_epochs = config.get('num_epochs', 15)

        # Estimate progress (assuming ~8322 steps per epoch for COCO)
        steps_per_epoch = 8322
        current_step = checkpoint.get('global_step', 0)
        current_epoch = current_step / steps_per_epoch

        print(f"\nEstimated Epoch: {current_epoch:.2f} / {num_epochs}")
        print(f"Progress: {current_epoch/num_epochs*100:.1f}%")

        print(f"\nConfig:")
        print(f"  freeze_bert: {config.get('freeze_bert', 'N/A')}")
        print(f"  learning_rate: {config.get('learning_rate', 'N/A')}")
        print(f"  contrastive_weight: {config.get('contrastive_weight', 'N/A')}")
        print(f"  recon_weight: {config.get('recon_weight', 'N/A')}")
        print(f"  diffusion_weight: {config.get('diffusion_weight', 'N/A')}")
else:
    print("No checkpoints found!")

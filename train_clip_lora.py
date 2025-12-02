"""
CLIP LoRA Fine-tuning with Flickr30k Dataset

This script fine-tunes a pre-trained CLIP model using LoRA (Low-Rank Adaptation)
on the Flickr30k dataset, ensuring every image has a corresponding text caption.

Key changes from previous version:
1. Uses Flickr30kDataset from main.py (same as step2.py) for reliable image-text pairs
2. Ensures all images have corresponding captions, preventing training errors
3. Batch format: dict with 'image' and 'caption' keys (compatible with step2.py)
4. Uses CLIP's own tokenizer for text encoding

This approach guarantees that:
- Every image has a valid text description
- No missing text issues that could break training
- Data loading is consistent with other training scripts (step2.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import time

# Import from main.py for Flickr30k dataset (same as step2.py)
# This ensures we use the same image-text pairing mechanism
from main import (
    Flickr30kDataset,
    SimpleTokenizer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load CLIP model
try:
    import clip
except ImportError:
    logger.error("CLIP not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    import clip


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning
    Uses float32 internally for numerical stability, converts to match input dtype
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices - ALWAYS stored in float32 for stability
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.float32))

        # Initialize both matrices with small non-zero values
        # This ensures gradients can flow properly from the start
        # Following the standard LoRA initialization from Microsoft's paper
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        # Initialize lora_B with very small values (not zero!) for stable training
        nn.init.normal_(self.lora_B, mean=0.0, std=0.01)

    def forward(self, x):
        # LoRA computation in float32, then convert to input dtype
        # This provides numerical stability during training
        original_dtype = x.dtype

        # Convert input to float32 for LoRA computation
        x_fp32 = x.to(dtype=torch.float32)

        # Compute LoRA update in float32
        lora_output = (x_fp32 @ self.lora_A @ self.lora_B) * self.scaling

        # Convert back to original dtype
        return lora_output.to(dtype=original_dtype)



def inject_lora_to_linear(model, target_modules=['q_proj', 'v_proj'], rank=8, alpha=16.0):
    lora_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not any(t in name for t in target_modules):
                continue

            in_features = module.in_features
            out_features = module.out_features

            # get device of the original module
            param = next(module.parameters())
            device = param.device
            model_dtype = param.dtype

            # create LoRA - it will use float32 internally
            lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
            # Only move to device, keep float32 dtype
            lora = lora.to(device=device)

            # patch forward
            orig_forward = module.forward

            def new_forward(x, orig_forward=orig_forward, l=lora):
                return orig_forward(x) + l(x)

            module.forward = new_forward
            module.add_module("lora_layer", lora)
            lora_layers.append(lora)

            logger.info(
                f"Injected LoRA into {name} (in={in_features}, out={out_features}, "
                f"rank={rank}, alpha={alpha}, scaling={lora.scaling:.3f}, "
                f"lora_dtype=float32, model_dtype={model_dtype}, device={device})"
            )

    return lora_layers


class CLIPLoRATrainer:
    """
    Trainer for fine-tuning CLIP with LoRA
    """

    def __init__(
        self,
        model_name='ViT-B/32',
        rank=8,
        alpha=8.0,
        target_modules=['q_proj', 'v_proj', 'c_fc'],
        device='cuda'
    ):
        self.device = device
        self.rank = rank
        self.alpha = alpha

        # Load CLIP model
        logger.info(f'Loading CLIP model: {model_name}')
        self.model, self.preprocess = clip.load(model_name, device=device)

        # Freeze original parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject LoRA layers
        logger.info(f'Injecting LoRA layers (rank={rank}, alpha={alpha})')
        self.lora_layers = inject_lora_to_linear(
            self.model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha
        )

        # Make LoRA parameters trainable
        self.trainable_params = []
        for _, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if 'lora' in param_name.lower():
                    param.requires_grad = True
                    self.trainable_params.append(param)

        logger.info(f'Number of LoRA layers: {len(self.lora_layers)}')
        logger.info(f'Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}')

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f'Total parameters: {total_params:,}')
        logger.info(f'Trainable ratio: {sum(p.numel() for p in self.trainable_params)/total_params*100:.2f}%')

    def compute_clip_loss(self, image_features, text_features, logit_scale):
        """
        Compute contrastive loss for CLIP with numerical stability checks
        Returns: (loss, logits_per_image) for accuracy computation
        """
        # Check for NaN or Inf in features before normalization
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            logger.error(f"NaN or Inf detected in image_features before normalization!")
            logger.error(f"image_features stats - min: {image_features.min()}, max: {image_features.max()}, mean: {image_features.mean()}")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            logger.error(f"NaN or Inf detected in text_features before normalization!")
            logger.error(f"text_features stats - min: {text_features.min()}, max: {text_features.max()}, mean: {text_features.mean()}")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        # Normalize features with eps for numerical stability
        image_features = F.normalize(image_features, dim=-1, eps=1e-6)
        text_features = F.normalize(text_features, dim=-1, eps=1e-6)

        # Check for NaN or Inf after normalization
        if torch.isnan(image_features).any() or torch.isinf(image_features).any():
            logger.error(f"NaN or Inf detected in image_features after normalization!")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            logger.error(f"NaN or Inf detected in text_features after normalization!")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        # Clamp logit_scale to prevent overflow (max ~4.6 corresponds to scale ~100)
        logit_scale = torch.clamp(logit_scale, max=4.6052)  # log(100)

        # Compute logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Check logits for NaN or Inf
        if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
            logger.error(f"NaN or Inf detected in logits! logit_scale: {logit_scale.item()}")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        # Labels: diagonal is positive pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        # Contrastive loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        # Final check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN or Inf detected in final loss!")
            logger.error(f"loss_img: {loss_img.item()}, loss_txt: {loss_txt.item()}")
            return torch.tensor(float('nan'), device=self.device, requires_grad=True), None

        return loss, logits_per_image

    def compute_accuracy(self, logits):
        """
        Compute retrieval accuracy (same as main.py)
        Args:
            logits: (batch_size, batch_size) similarity matrix
        Returns:
            dict with i2t_top1, i2t_top5, t2i_top1, t2i_top5
        """
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        # Adjust k for top-k based on batch size
        k = min(5, batch_size)

        # Image to text retrieval (top-1 and top-k)
        _, i2t_pred = logits.topk(k, dim=1)
        i2t_top1 = (i2t_pred[:, 0] == labels).float().mean().item()
        i2t_topk = (i2t_pred == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        # Text to image retrieval (top-1 and top-k)
        _, t2i_pred = logits.t().topk(k, dim=1)
        t2i_top1 = (t2i_pred[:, 0] == labels).float().mean().item()
        t2i_topk = (t2i_pred == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        return {
            'i2t_top1': i2t_top1,
            'i2t_top5': i2t_topk,
            't2i_top1': t2i_top1,
            't2i_top5': t2i_topk,
            'avg_top1': (i2t_top1 + t2i_top1) / 2,
            'avg_top5': (i2t_topk + t2i_topk) / 2
        }

    @torch.no_grad()
    def evaluate(self, val_loader):
        """
        Evaluate model on validation set
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0

        # Accuracy accumulators
        total_i2t_top1 = 0
        total_i2t_top5 = 0
        total_t2i_top1 = 0
        total_t2i_top5 = 0

        pbar = tqdm(val_loader, desc='Validating', ncols=120)

        for batch in pbar:
            images = batch['image'].to(self.device)
            captions = batch['caption']

            # Check input images for NaN or Inf
            if torch.isnan(images).any() or torch.isinf(images).any():
                logger.warning(f"Skipping batch with NaN/Inf in images during validation")
                continue

            # Tokenize text using CLIP's tokenizer
            text_tokens = clip.tokenize(captions, truncate=True).to(self.device)

            # Forward pass
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)

            # Compute loss
            logit_scale = self.model.logit_scale.exp()
            loss, logits = self.compute_clip_loss(image_features, text_features, logit_scale)

            if logits is None:
                continue

            # Compute accuracy
            accuracy = self.compute_accuracy(logits)
            total_i2t_top1 += accuracy['i2t_top1']
            total_i2t_top5 += accuracy['i2t_top5']
            total_t2i_top1 += accuracy['t2i_top1']
            total_t2i_top5 += accuracy['t2i_top5']

            total_loss += loss.item()
            total_samples += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate validation metrics
        num_batches = len(val_loader)
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_i2t_top1': total_i2t_top1 / num_batches,
            'val_i2t_top5': total_i2t_top5 / num_batches,
            'val_t2i_top1': total_t2i_top1 / num_batches,
            'val_t2i_top5': total_t2i_top5 / num_batches,
        }

        logger.info(f"Validation Results:")
        logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Image->Text: Top-1={val_metrics['val_i2t_top1']*100:.2f}%, Top-5={val_metrics['val_i2t_top5']*100:.2f}%")
        logger.info(f"  Text->Image: Top-1={val_metrics['val_t2i_top1']*100:.2f}%, Top-5={val_metrics['val_t2i_top5']*100:.2f}%")

        return val_metrics

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs=10,
        learning_rate=1e-4,
        save_dir='outputs/clip_lora',
        save_freq=5
    ):
        """
        Train CLIP with LoRA (with validation support like main.py)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Optimizer only for LoRA parameters with more conservative settings
        optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8  # Increase epsilon for numerical stability
        )

        # Learning rate scheduler (ReduceLROnPlateau like main.py)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )

        logger.info(f'Starting training for {num_epochs} epochs...')
        logger.info(f'Total batches per epoch: {len(train_loader)}')
        if val_loader:
            logger.info(f'Validation batches per epoch: {len(val_loader)}')

        best_val_loss = float('inf')
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            epoch_start_time = time.time()

            # Accuracy accumulators for training
            total_i2t_top1 = 0
            total_i2t_top5 = 0
            total_t2i_top1 = 0
            total_t2i_top5 = 0

            pbar = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{num_epochs}',
                ncols=150
            )

            batch_start_time = time.time()

            for batch_idx, batch in enumerate(pbar):
                # Get images and captions from batch dict (same as step2.py)
                images = batch['image'].to(self.device)
                captions = batch['caption']  # List of strings

                # Check input images for NaN or Inf
                if torch.isnan(images).any() or torch.isinf(images).any():
                    logger.error(f"Batch {batch_idx}: NaN or Inf detected in input images!")
                    logger.error(f"Images stats - min: {images.min()}, max: {images.max()}, mean: {images.mean()}")
                    continue

                # Tokenize text using CLIP's tokenizer
                text_tokens = clip.tokenize(captions, truncate=True).to(self.device)

                # Forward pass
                image_features = self.model.encode_image(images)

                # Check image features immediately after encoding
                if torch.isnan(image_features).any() or torch.isinf(image_features).any():
                    logger.error(f"Batch {batch_idx}: NaN/Inf in image_features after encode_image")
                    logger.error(f"  image_features stats - min: {image_features.min()}, max: {image_features.max()}")
                    # Check LoRA parameters
                    for i, param in enumerate(self.trainable_params[:5]):
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.error(f"  LoRA param {i} has NaN/Inf!")
                    continue

                text_features = self.model.encode_text(text_tokens)

                # Compute loss and logits
                logit_scale = self.model.logit_scale.exp()
                loss, logits = self.compute_clip_loss(image_features, text_features, logit_scale)

                if logits is None:
                    continue

                # Compute accuracy
                accuracy = self.compute_accuracy(logits)
                total_i2t_top1 += accuracy['i2t_top1']
                total_i2t_top5 += accuracy['i2t_top5']
                total_t2i_top1 += accuracy['t2i_top1']
                total_t2i_top5 += accuracy['t2i_top5']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Check for NaN gradients (with float32 this should be rare)
                nan_gradients = False
                max_grad = 0.0
                for i, param in enumerate(self.trainable_params):
                    if param.grad is not None:
                        grad_abs_max = param.grad.abs().max().item()
                        max_grad = max(max_grad, grad_abs_max)

                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.error(f"Batch {batch_idx}: NaN or Inf gradient in parameter {i}")
                            logger.error(f"Grad stats - min: {param.grad.min()}, max: {param.grad.max()}, mean: {param.grad.mean()}")
                            nan_gradients = True
                            break

                if nan_gradients:
                    logger.error(f"Batch {batch_idx}: Skipping optimizer step due to NaN gradients")
                    continue

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)

                # Check if gradient norm is NaN
                if torch.isnan(grad_norm):
                    logger.error(f"Batch {batch_idx}: NaN gradient norm, skipping update")
                    continue

                # Update parameters
                optimizer.step()

                total_loss += loss.item()

                # Calculate speed and ETA (like main.py)
                batch_time = time.time() - batch_start_time
                batch_size = images.size(0)
                samples_per_sec = batch_size / batch_time if batch_time > 0 else 0

                batches_done = batch_idx + 1
                batches_left = len(train_loader) - batches_done
                eta_seconds = batches_left * (time.time() - epoch_start_time) / batches_done
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)

                # Update progress bar with detailed info
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/batches_done:.4f}',
                    'i2t': f'{accuracy["i2t_top1"]*100:.1f}%',
                    't2i': f'{accuracy["t2i_top1"]*100:.1f}%',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'speed': f'{samples_per_sec:.1f}s/s',
                    'ETA': f'{eta_min:02d}:{eta_sec:02d}'
                })

                batch_start_time = time.time()

            # Calculate epoch metrics
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_i2t_top1 = total_i2t_top1 / num_batches
            avg_i2t_top5 = total_i2t_top5 / num_batches
            avg_t2i_top1 = total_t2i_top1 / num_batches
            avg_t2i_top5 = total_t2i_top5 / num_batches

            # Print separator for clarity
            logger.info('')
            logger.info('='*80)
            logger.info(f'EPOCH {epoch+1}/{num_epochs} SUMMARY')
            logger.info('='*80)

            logger.info(f'\n[Training Results]')
            logger.info(f'  Loss: {avg_loss:.4f}')
            logger.info(f'  Image->Text: Top-1={avg_i2t_top1*100:.2f}%, Top-5={avg_i2t_top5*100:.2f}%')
            logger.info(f'  Text->Image: Top-1={avg_t2i_top1*100:.2f}%, Top-5={avg_t2i_top5*100:.2f}%')
            logger.info(f'  Average Accuracy: Top-1={(avg_i2t_top1+avg_t2i_top1)/2*100:.2f}%')

            # Validation - 每个epoch都运行
            if val_loader:
                logger.info(f'\n[Validation Results]')
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['val_loss']

                logger.info(f'  Loss: {val_loss:.4f} (Δ {val_loss-avg_loss:+.4f} vs train)')
                logger.info(f'  Image->Text: Top-1={val_metrics["val_i2t_top1"]*100:.2f}%, Top-5={val_metrics["val_i2t_top5"]*100:.2f}%')
                logger.info(f'  Text->Image: Top-1={val_metrics["val_t2i_top1"]*100:.2f}%, Top-5={val_metrics["val_t2i_top5"]*100:.2f}%')
                val_avg_acc = (val_metrics['val_i2t_top1'] + val_metrics['val_t2i_top1']) / 2
                logger.info(f'  Average Accuracy: Top-1={val_avg_acc*100:.2f}%')

                # Update scheduler based on validation loss
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                # Log learning rate changes
                logger.info(f'\n[Training Info]')
                if new_lr < old_lr:
                    logger.info(f'  Learning rate: {old_lr:.2e} -> {new_lr:.2e} (REDUCED)')
                else:
                    logger.info(f'  Learning rate: {new_lr:.2e}')

                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    best_checkpoint_path = save_dir / 'clip_lora_best.pt'
                    self.save_checkpoint(best_checkpoint_path, epoch + 1, optimizer, scheduler, val_loss)
                    logger.info(f'  ⭐ NEW BEST MODEL! Val loss improved by {improvement:.4f}')
                    logger.info(f'  Saved to: {best_checkpoint_path}')
                else:
                    logger.info(f'  Best val loss so far: {best_val_loss:.4f} (current: {val_loss:.4f})')

                # Also track best accuracy
                if val_avg_acc > best_val_acc:
                    best_val_acc = val_avg_acc
                    logger.info(f'  ⭐ NEW BEST ACCURACY! {val_avg_acc*100:.2f}%')
                else:
                    logger.info(f'  Best accuracy so far: {best_val_acc*100:.2f}%')

                # Performance summary table
                logger.info(f'\n[Performance Comparison]')
                logger.info(f'  {"Metric":<25} {"Train":>12} {"Validation":>12} {"Difference":>12}')
                logger.info(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*12}')
                logger.info(f'  {"Loss":<25} {avg_loss:>12.4f} {val_loss:>12.4f} {val_loss-avg_loss:>+12.4f}')
                logger.info(f'  {"I2T Top-1":<25} {avg_i2t_top1*100:>11.2f}% {val_metrics["val_i2t_top1"]*100:>11.2f}% {(val_metrics["val_i2t_top1"]-avg_i2t_top1)*100:>+11.2f}%')
                logger.info(f'  {"T2I Top-1":<25} {avg_t2i_top1*100:>11.2f}% {val_metrics["val_t2i_top1"]*100:>11.2f}% {(val_metrics["val_t2i_top1"]-avg_t2i_top1)*100:>+11.2f}%')
                logger.info(f'  {"Average Top-1":<25} {(avg_i2t_top1+avg_t2i_top1)/2*100:>11.2f}% {val_avg_acc*100:>11.2f}% {(val_avg_acc-(avg_i2t_top1+avg_t2i_top1)/2)*100:>+11.2f}%')

            else:
                # No validation, just update scheduler
                logger.info(f'\n[Training Info]')
                logger.info(f'  No validation set provided')
                scheduler.step(avg_loss)
                logger.info(f'  Learning rate: {optimizer.param_groups[0]["lr"]:.2e}')

            logger.info('='*80)
            logger.info('')

            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                checkpoint_path = save_dir / f'clip_lora_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path, epoch + 1, optimizer, scheduler, avg_loss)
                logger.info(f'Checkpoint saved to {checkpoint_path}')

        # Save final model
        final_path = save_dir / 'clip_lora_final.pt'
        self.save_checkpoint(final_path, num_epochs, optimizer, scheduler, avg_loss)
        logger.info(f'Final model saved to {final_path}')

    def save_checkpoint(self, path, epoch, optimizer, scheduler, loss):
        """
        Save model checkpoint with only LoRA parameters
        """
        # Save only LoRA parameters to keep checkpoint small
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower() or param.requires_grad:
                lora_state_dict[name] = param.cpu()

        torch.save({
            'epoch': epoch,
            'lora_state_dict': lora_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'rank': self.rank,
            'alpha': self.alpha,
        }, path)

    def load_checkpoint(self, path):
        """
        Load LoRA checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load LoRA parameters
        lora_state_dict = checkpoint['lora_state_dict']
        model_state_dict = self.model.state_dict()

        # Update only LoRA parameters
        for name, param in lora_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name] = param.to(self.device)

        self.model.load_state_dict(model_state_dict)
        logger.info(f'Loaded LoRA checkpoint from {path}')
        logger.info(f'Epoch: {checkpoint["epoch"]}, Loss: {checkpoint["loss"]:.4f}')

    def encode_text(self, text_prompts):
        """
        Encode text prompts to embeddings
        For use in step2_lora.py (text-to-image diffusion)

        Args:
            text_prompts: List of text strings or pre-tokenized tensor

        Returns:
            text_embeddings: (batch_size, 512) tensor
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize if needed
            if isinstance(text_prompts, list):
                text_tokens = clip.tokenize(text_prompts, truncate=True).to(self.device)
            else:
                text_tokens = text_prompts.to(self.device)

            # Get text features from CLIP
            text_features = self.model.encode_text(text_tokens)

            # Normalize (important for CLIP)
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def get_text_encoder(self):
        """
        Get the text encoder for use in diffusion models
        Returns a wrapper that's compatible with step2.py's TextEncoder interface
        """
        return CLIPTextEncoderWrapper(self)


class CLIPTextEncoderWrapper(nn.Module):
    """
    Wrapper to make CLIP compatible with step2.py's TextEncoder interface
    step2.py expects: text_encoder(text_tokens) -> (batch_size, seq_len, embed_dim)
    CLIP provides: encode_text(text_tokens) -> (batch_size, 512)
    """

    def __init__(self, clip_trainer):
        super().__init__()
        self.clip_trainer = clip_trainer
        self.embed_dim = 512  # CLIP ViT-B/32 output dimension

    def forward(self, text_prompts):
        """
        Args:
            text_prompts: List of text strings or tokenized tensor
        Returns:
            text_embeddings: (batch_size, 1, 512) - added sequence dimension for CrossAttention
        """
        # Get CLIP text features
        text_features = self.clip_trainer.encode_text(text_prompts)

        # Add sequence dimension: (batch_size, 512) -> (batch_size, 1, 512)
        # step2.py's CrossAttention expects (batch_size, seq_len, embed_dim)
        text_embeddings = text_features.unsqueeze(1)

        return text_embeddings


def main():
    """
    Main training function - using Flickr30k dataset (same as step2.py)
    """
    config = {
        # Data config - using Flickr30k dataset for reliable image-text pairs
        'data_dir': 'data/flickr30k',  # Flickr30k dataset directory
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'image_size': 224,  # CLIP uses 224x224 images
        'save_dir': 'outputs/clip_lora',
        'save_freq': 1,

        # LoRA hyperparameters
        'lora_rank': 8,  # Rank of LoRA matrices (lower = fewer parameters)
        'lora_alpha': 8.0,  # LoRA scaling factor
        'target_modules': ['q_proj', 'v_proj', 'c_fc', 'out_proj'],  # Which layers to add LoRA to

        # CLIP model
        'clip_model': 'ViT-B/32',  # Options: ViT-B/32, ViT-B/16, ViT-L/14

        # Tokenizer config (needed for Flickr30kDataset)
        'vocab_size': 10000,
        'max_length': 77,
        'train_ratio': 0.9,  # Train/val split ratio
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize tokenizer (required by Flickr30kDataset)
    logger.info('Building tokenizer...')
    tokenizer = SimpleTokenizer(
        vocab_size=config['vocab_size'],
        max_length=config['max_length']
    )

    # Build vocab from all captions first
    temp_dataset = Flickr30kDataset(
        config['data_dir'],
        image_size=config['image_size'],
        split='all'
    )
    tokenizer.build_vocab(temp_dataset.get_all_captions())
    logger.info(f'Vocabulary built with {len(tokenizer.word2idx)} words')

    # Load training dataset with tokenizer (same as step2.py)
    logger.info(f'Loading dataset from {config["data_dir"]}...')

    # CLIP transform (same for train and val)
    clip_transform = transforms.Compose([
        transforms.Resize(config['image_size'], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    try:
        # Training dataset
        train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            tokenizer=tokenizer,
            split='train',
            train_ratio=config['train_ratio']
        )
        train_dataset.transform = clip_transform
        logger.info('Applied CLIP-specific image transforms to training set')

        # Validation dataset
        val_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config['image_size'],
            tokenizer=tokenizer,
            split='val',
            train_ratio=config['train_ratio']
        )
        val_dataset.transform = clip_transform
        logger.info('Applied CLIP-specific image transforms to validation set')

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=False
        )

        logger.info(f'Training dataset size: {len(train_dataset)}')
        logger.info(f'Validation dataset size: {len(val_dataset)}')
        logger.info(f'Training batches: {len(train_loader)}')
        logger.info(f'Validation batches: {len(val_loader)}')

    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        logger.info('Please make sure you have the Flickr30k dataset in data/flickr30k directory')
        return

    # Create trainer
    logger.info('Creating CLIP LoRA trainer...')
    trainer = CLIPLoRATrainer(
        model_name=config['clip_model'],
        rank=config['lora_rank'],
        alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        device=device
    )

    # Train
    logger.info('Starting training...')
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,  # 添加验证集
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir=config['save_dir'],
        save_freq=config['save_freq']
    )

    logger.info('Training completed!')
    logger.info(f'Best validation loss: {trainer.best_loss:.4f}' if hasattr(trainer, 'best_loss') else '')


def test_clip_lora(
    checkpoint_path='outputs/clip_lora/clip_lora_final.pt',
    test_image_path=None,
    test_texts=None,
    clip_model='ViT-B/32',
    lora_rank=8,
    lora_alpha=8.0,
    device='cuda'
):
    """
    Test CLIP with LoRA on a single image

    Args:
        checkpoint_path: Path to LoRA checkpoint
        test_image_path: Path to test image
        test_texts: List of text prompts to compare
        clip_model: CLIP model name
        lora_rank: LoRA rank used during training
        lora_alpha: LoRA alpha used during training
        device: Device to use
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Default test texts
    if test_texts is None:
        test_texts = [
            'a beautiful landscape photograph',
            'a mountain landscape',
            'a sunset over water',
            'a forest scene',
            'an urban cityscape'
        ]

    # Create trainer and load checkpoint
    logger.info('Loading CLIP with LoRA...')
    trainer = CLIPLoRATrainer(
        model_name=clip_model,
        rank=lora_rank,
        alpha=lora_alpha,
        device=device
    )
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()

    # Load and preprocess test image
    if test_image_path is None:
        logger.error('Please provide a test image path')
        return

    logger.info(f'Loading test image: {test_image_path}')
    image = Image.open(test_image_path).convert('RGB')

    # Transform
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(test_texts).to(device)

    # Compute features
    with torch.no_grad():
        image_features = trainer.model.encode_image(image_tensor)
        text_features = trainer.model.encode_text(text_tokens)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        probs = F.softmax(similarities * 100, dim=0)

    # Print results
    logger.info('\nResults:')
    logger.info('-' * 50)
    for text, prob in zip(test_texts, probs):
        logger.info(f'{text:<40} {prob.item()*100:>6.2f}%')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CLIP LoRA Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--checkpoint', type=str, default='outputs/clip_lora/clip_lora_final.pt',
                       help='Checkpoint path for testing')
    parser.add_argument('--test-image', type=str, default=None,
                       help='Test image path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        test_clip_lora(
            checkpoint_path=args.checkpoint,
            test_image_path=args.test_image,
            device=args.device
        )

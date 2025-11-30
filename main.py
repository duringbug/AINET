import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Simple tokenizer for text processing"""

    def __init__(self, vocab_size=10000, max_length=77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_count[word] = self.word_count.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)

        # Add most frequent words to vocabulary
        for word, _ in sorted_words[:self.vocab_size - 4]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(f"Vocabulary built with {len(self.word2idx)} words")

    def encode(self, text):
        """Encode text to token IDs"""
        words = text.lower().split()
        tokens = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return tokens

    def batch_encode(self, texts):
        """Encode batch of texts"""
        return [self.encode(text) for text in texts]


class Flickr30kDataset(Dataset):
    """Flickr30k image-text dataset"""

    def __init__(self, data_dir, image_size=224, tokenizer=None):
        """
        Args:
            data_dir: Dataset root directory
            image_size: Image size for resizing
            tokenizer: Tokenizer instance
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer

        # Find image directory
        possible_image_dirs = [
            self.data_dir / 'flickr30k_images' / 'flickr30k_images',
            self.data_dir / 'flickr30k_images',
            self.data_dir / 'Images',
            self.data_dir / 'images',
        ]

        self.image_dir = None
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                self.image_dir = img_dir
                logger.info(f"Found image directory: {img_dir}")
                break

        if self.image_dir is None:
            raise ValueError(f"Image directory not found in {self.data_dir}")

        # Load captions
        self.captions = self._load_captions()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info(f"Dataset initialized with {len(self.captions)} samples")

    def _load_captions(self):
        """Load image captions"""
        # Try multiple possible locations for caption files
        caption_files = [
            self.data_dir / 'results.csv',
            self.data_dir / 'flickr30k_images' / 'results.csv',
            self.data_dir / 'captions.txt',
            self.data_dir / 'flickr30k_images' / 'captions.txt',
            self.data_dir / 'results_20130124.token',
        ]

        captions = []

        # Try loading CSV format
        for caption_file in caption_files:
            if caption_file.exists():
                logger.info(f"Loading captions from: {caption_file}")
                if caption_file.suffix == '.csv':
                    try:
                        # Try with pipe separator first (Flickr30k format)
                        df = pd.read_csv(caption_file, sep='|')
                        df.columns = df.columns.str.strip()  # Remove whitespace from column names

                        # Check different possible column names
                        img_col = None
                        caption_col = None

                        if 'image_name' in df.columns:
                            img_col = 'image_name'
                        elif 'image' in df.columns:
                            img_col = 'image'

                        if 'comment' in df.columns:
                            caption_col = 'comment'
                        elif 'caption' in df.columns:
                            caption_col = 'caption'

                        if img_col and caption_col:
                            logger.info(f"Found columns: {img_col}, {caption_col}")
                            for _, row in df.iterrows():
                                img_name = str(row[img_col]).strip()
                                img_path = self.image_dir / img_name
                                if img_path.exists():
                                    captions.append({
                                        'image': str(img_path),
                                        'caption': str(row[caption_col]).strip()
                                    })
                            if captions:
                                logger.info(f"Loaded {len(captions)} image-caption pairs")
                                return captions
                    except Exception as e:
                        logger.warning(f"Failed to load {caption_file} with pipe separator: {e}")
                        # Try with comma separator
                        try:
                            df = pd.read_csv(caption_file)
                            if 'image' in df.columns and 'caption' in df.columns:
                                for _, row in df.iterrows():
                                    img_path = self.image_dir / row['image']
                                    if img_path.exists():
                                        captions.append({
                                            'image': str(img_path),
                                            'caption': str(row['caption'])
                                        })
                                if captions:
                                    return captions
                        except:
                            pass
                elif caption_file.suffix == '.txt':
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '|' in line:
                                parts = line.strip().split('|')
                                if len(parts) >= 2:
                                    img_name = parts[0].strip()
                                    caption = parts[1].strip()
                                    img_path = self.image_dir / img_name
                                    if img_path.exists():
                                        captions.append({
                                            'image': str(img_path),
                                            'caption': caption
                                        })
                    if captions:
                        return captions

        # If no caption file found, create simple captions
        logger.warning("No caption file found, creating simple captions")
        image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        for img_path in image_files[:1000]:
            captions.append({
                'image': str(img_path),
                'caption': 'an image from flickr30k dataset'
            })

        return captions

    def get_all_captions(self):
        """Get all captions for vocabulary building"""
        return [item['caption'] for item in self.captions]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.captions[idx]

        # Load image
        try:
            image = Image.open(item['image']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {item['image']}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)

        caption = item['caption']

        # Tokenize caption
        if self.tokenizer:
            tokens = self.tokenizer.encode(caption)
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = None

        return {
            'image': image,
            'caption': caption,
            'tokens': tokens,
            'image_path': item['image']
        }


class SimpleCNN(nn.Module):
    """Lightweight CNN for image encoding"""

    def __init__(self, embed_dim=256):
        super().__init__()

        # Simpler convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Conv block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv block 5: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simpler projection
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        features = self.conv_layers(x)  # (batch_size, 256, 7, 7)
        features = self.global_pool(features)  # (batch_size, 256, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size, 256)
        embeddings = self.projection(features)  # (batch_size, embed_dim)
        return embeddings


class ImageEncoder(nn.Module):
    """Image encoder with option for CNN or ResNet"""

    def __init__(self, embed_dim=512, use_simple_cnn=True):
        super().__init__()

        if use_simple_cnn:
            logger.info("Using Simple CNN for image encoding")
            self.encoder = SimpleCNN(embed_dim)
        else:
            logger.info("Using ResNet50 for image encoding")
            # Use pretrained ResNet50 as backbone
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            resnet_dim = 2048
            self.projection = nn.Sequential(
                nn.Linear(resnet_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.encoder = None

    def forward(self, x):
        if self.encoder is not None:
            # Simple CNN
            return self.encoder(x)
        else:
            # ResNet
            features = self.backbone(x)
            features = features.squeeze(-1).squeeze(-1)
            embeddings = self.projection(features)
            return embeddings


class TextEncoder(nn.Module):
    """Lightweight text encoder using LSTM"""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=1):
        super().__init__()

        self.embed_dim = embed_dim

        # Word embedding (smaller dimension)
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        # LSTM encoder (single layer, bidirectional)
        self.lstm = nn.LSTM(
            128,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Simpler projection
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, 128)

        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embeddings)

        # Use the last hidden state (concatenate forward and backward)
        # hidden: (num_layers * 2, batch_size, hidden_dim)
        forward_hidden = hidden[-2]  # (batch_size, hidden_dim)
        backward_hidden = hidden[-1]  # (batch_size, hidden_dim)
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Project to embed_dim
        text_features = self.projection(final_hidden)  # (batch_size, embed_dim)

        return text_features


class MultimodalModel(nn.Module):
    """Multimodal model for image-text matching"""

    def __init__(self, vocab_size, embed_dim=512, use_simple_cnn=True):
        super().__init__()

        self.embed_dim = embed_dim

        # Image and text encoders
        self.image_encoder = ImageEncoder(embed_dim, use_simple_cnn=use_simple_cnn)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)

        # Temperature parameter for contrastive learning (higher for easier training)
        self.temperature = nn.Parameter(torch.ones([]) * 0.1)

    def forward(self, images, tokens):
        # Encode images and text
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(tokens)

        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

    def compute_loss(self, image_features, text_features):
        """Compute contrastive learning loss (InfoNCE)"""

        # Cosine similarity as logits
        # logits: (batch_size, batch_size)
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric cross-entropy loss
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)

        loss = (loss_i2t + loss_t2i) / 2

        return loss, logits


class Trainer:
    """Trainer for multimodal model"""

    def __init__(self, config):
        self.config = config

        # Setup device - prioritize MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon GPU) acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Using CUDA GPU acceleration")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        self.tokenizer = SimpleTokenizer(
            vocab_size=config.get('vocab_size', 10000),
            max_length=config.get('max_text_length', 77)
        )

        # Load dataset for vocabulary building
        temp_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config.get('image_size', 224)
        )

        # Build vocabulary
        logger.info("Building vocabulary from captions...")
        self.tokenizer.build_vocab(temp_dataset.get_all_captions())

        # Load dataset with tokenizer
        self.train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config.get('image_size', 224),
            tokenizer=self.tokenizer
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=False  # Disable for MPS
        )

        # Initialize model
        self.model = MultimodalModel(
            vocab_size=len(self.tokenizer.word2idx),
            embed_dim=config.get('embed_dim', 512),
            use_simple_cnn=config.get('use_simple_cnn', True)
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        # Training statistics
        self.global_step = 0
        self.best_loss = float('inf')

    def compute_accuracy(self, logits):
        """Compute retrieval accuracy"""
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

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        epoch_start_time = time.time()

        # Accuracy accumulators
        total_i2t_top1 = 0
        total_i2t_topk = 0
        total_t2i_top1 = 0
        total_t2i_topk = 0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}',
            ncols=150
        )

        batch_start_time = time.time()

        for batch_idx, batch in enumerate(pbar):
            # Prepare data
            images = batch['image'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            batch_size = images.size(0)

            # Forward pass
            image_features, text_features = self.model(images, tokens)

            # Compute loss
            loss, logits = self.model.compute_loss(image_features, text_features)

            # Compute accuracy
            accuracy = self.compute_accuracy(logits)
            total_i2t_top1 += accuracy['i2t_top1']
            total_i2t_topk += accuracy['i2t_top5']
            total_t2i_top1 += accuracy['t2i_top1']
            total_t2i_topk += accuracy['t2i_top5']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_samples += batch_size
            self.global_step += 1

            # Calculate speed
            batch_time = time.time() - batch_start_time
            samples_per_sec = batch_size / batch_time if batch_time > 0 else 0

            # Calculate ETA
            batches_done = batch_idx + 1
            batches_left = len(self.train_loader) - batches_done
            eta_seconds = batches_left * (time.time() - epoch_start_time) / batches_done
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)

            # Update progress bar with detailed info
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batches_done:.4f}',
                'i2t': f'{accuracy["i2t_top1"]*100:.1f}%',
                't2i': f'{accuracy["t2i_top1"]*100:.1f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'speed': f'{samples_per_sec:.1f}s/s',
                'ETA': f'{eta_min:02d}:{eta_sec:02d}'
            })

            batch_start_time = time.time()

            # Save checkpoint periodically
            if self.global_step % self.config.get('save_steps', 500) == 0:
                self.save_checkpoint(f'checkpoint-step-{self.global_step}')

        # Calculate epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        avg_i2t_top1 = total_i2t_top1 / len(self.train_loader)
        avg_i2t_topk = total_i2t_topk / len(self.train_loader)
        avg_t2i_top1 = total_t2i_top1 / len(self.train_loader)
        avg_t2i_topk = total_t2i_topk / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch+1} Summary:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Image->Text: Top-1={avg_i2t_top1*100:.2f}%, Top-K={avg_i2t_topk*100:.2f}%")
        logger.info(f"  Text->Image: Top-1={avg_t2i_top1*100:.2f}%, Top-K={avg_t2i_topk*100:.2f}%")
        logger.info(f"  Time: {epoch_time:.1f}s ({total_samples/epoch_time:.1f} samples/s)")

        return avg_loss

    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config['num_epochs']}")
        logger.info(f"Batch size: {self.config['batch_size']}")
        logger.info(f"Total steps per epoch: {len(self.train_loader)}")

        for epoch in range(self.config['num_epochs']):
            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint('best_model')
                logger.info(f"New best model saved with loss: {avg_loss:.4f}")

            # Save checkpoint per epoch
            self.save_checkpoint(f'epoch-{epoch+1}')

        logger.info("Training completed!")
        logger.info(f"Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, name):
        """Save checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'tokenizer': {
                'word2idx': self.tokenizer.word2idx,
                'idx2word': self.tokenizer.idx2word,
                'vocab_size': self.tokenizer.vocab_size,
                'max_length': self.tokenizer.max_length
            }
        }, checkpoint_dir / 'pytorch_model.bin')

        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main function"""
    # Configuration
    config = {
        # Data config
        'data_dir': 'data/flickr30k',
        'output_dir': 'outputs',

        # Model config
        'vocab_size': 10000,
        'embed_dim': 256,  # Reduced from 512
        'image_size': 224,
        'max_text_length': 77,
        'use_simple_cnn': True,  # True: Simple CNN, False: ResNet50

        # Training config
        'num_epochs': 20,  # More epochs for better convergence
        'batch_size': 64,  # Larger batch size for contrastive learning
        'learning_rate': 5e-4,  # Higher learning rate
        'weight_decay': 0.01,
        'num_workers': 4,

        # Save config
        'save_steps': 500,
    }

    # Check data directory
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please run download.py first to download the dataset")
        return

    # Create trainer
    trainer = Trainer(config)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint('interrupted')
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

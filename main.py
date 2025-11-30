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
import os
from transformers import BertModel, BertTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Flickr30kDataset(Dataset):
    """Flickr30k image-text dataset"""

    def __init__(self, data_dir, image_size=224, tokenizer=None, split='train', train_ratio=0.9, random_seed=42):
        """
        Args:
            data_dir: Dataset root directory
            image_size: Image size for resizing
            tokenizer: Tokenizer instance
            split: 'train', 'val', or 'all'
            train_ratio: Ratio of data to use for training (default 0.9)
            random_seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed

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
        all_captions = self._load_captions()

        # Split dataset
        if split in ['train', 'val']:
            import random
            random.seed(random_seed)

            # Shuffle with fixed seed for reproducibility
            indices = list(range(len(all_captions)))
            random.shuffle(indices)

            # Split indices
            train_size = int(len(indices) * train_ratio)

            if split == 'train':
                selected_indices = indices[:train_size]
                logger.info(f"Using {len(selected_indices)} samples for training ({train_ratio*100:.1f}%)")
            else:  # val
                selected_indices = indices[train_size:]
                logger.info(f"Using {len(selected_indices)} samples for validation ({(1-train_ratio)*100:.1f}%)")

            self.captions = [all_captions[i] for i in selected_indices]
        else:  # all
            self.captions = all_captions
            logger.info(f"Using all {len(self.captions)} samples")

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

        # Tokenize with BERT
        if self.tokenizer:
            encoding = self.tokenizer.encode_plus(
                caption,
                add_special_tokens=True,
                max_length=self.tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            tokens = input_ids
        else:
            max_len = 77
            tokens = torch.zeros(max_len, dtype=torch.long)
            input_ids = tokens
            attention_mask = torch.zeros_like(tokens)

        return {
            'image': image,
            'caption': caption,
            'tokens': tokens,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
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
    """Text encoder using pre-trained BERT model"""

    def __init__(self, embed_dim=256, bert_model_name='bert-base-uncased', freeze_bert=False, cache_dir='./models/bert_cache'):
        super().__init__()

        self.embed_dim = embed_dim

        # Load pre-trained BERT model with cache support
        logger.info(f"Loading BERT model: {bert_model_name}")
        os.makedirs(cache_dir, exist_ok=True)

        self.bert = BertModel.from_pretrained(
            bert_model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )

        # Optionally freeze BERT parameters for faster training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # BERT output dimension is 768 for bert-base
        bert_dim = self.bert.config.hidden_size

        # Projection layer to map BERT output to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len) - token IDs from BERT tokenizer
            attention_mask: (batch_size, seq_len) - attention mask
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, bert_dim)

        # Project to target embedding dimension
        text_features = self.projection(cls_output)  # (batch_size, embed_dim)

        return text_features


class ImageDecoder(nn.Module):
    """Image decoder - reconstructs images from embeddings"""

    def __init__(self, embed_dim=512, image_size=224):
        super().__init__()

        self.image_size = image_size

        # Project embedding to spatial features
        self.projection = nn.Linear(embed_dim, 256 * 7 * 7)

        # Transposed convolution layers (reverse of encoder)
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, embeddings):
        # embeddings: (batch_size, embed_dim)
        batch_size = embeddings.size(0)

        # Project to spatial features
        features = self.projection(embeddings)  # (batch_size, 256*7*7)
        features = features.view(batch_size, 256, 7, 7)  # (batch_size, 256, 7, 7)

        # Decode to image
        images = self.decoder(features)  # (batch_size, 3, 224, 224)

        return images


class TextDecoder(nn.Module):
    """Text decoder - generates text from embeddings"""

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=256, num_layers=2, max_length=77):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        # Project embedding to LSTM hidden state
        self.embed_projection = nn.Linear(embed_dim, hidden_dim * num_layers)

        # Word embedding for decoder
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, embeddings, target_tokens=None, teacher_forcing_ratio=0.5):
        """
        Args:
            embeddings: (batch_size, embed_dim)
            target_tokens: (batch_size, seq_len) - for training with teacher forcing
            teacher_forcing_ratio: probability of using teacher forcing
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Initialize hidden state from embeddings
        hidden = self.embed_projection(embeddings)  # (batch_size, hidden_dim * num_layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, batch_size, hidden_dim)
        cell = torch.zeros_like(hidden)

        # Start with <SOS> token (token id 2)
        decoder_input = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)

        outputs = []

        for t in range(self.max_length):
            # Embed current input
            embedded = self.word_embedding(decoder_input)  # (batch_size, 1, hidden_dim)

            # LSTM step
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))

            # Project to vocabulary
            output = self.output_projection(lstm_out)  # (batch_size, 1, vocab_size)
            outputs.append(output)

            # Decide next input
            if target_tokens is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                if t < target_tokens.size(1) - 1:
                    decoder_input = target_tokens[:, t+1:t+2]
                else:
                    decoder_input = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
            else:
                # Use predicted token
                decoder_input = output.argmax(dim=-1)

        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_length, vocab_size)

        return outputs


class DiffusionModel(nn.Module):
    """Diffusion model for generating embeddings in the unified vector space"""

    def __init__(self, embed_dim=512, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Linear beta schedule - register as buffers so they move with the model
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register as buffers (will be moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim + 1, 512),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_diffusion(self, x_0, t):
        """Add noise to embeddings (forward diffusion process)"""
        # x_0: (batch_size, embed_dim)
        # t: (batch_size,) timestep indices

        # Buffers are already on the correct device
        # Get noise schedule values
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])

        # Sample noise
        noise = torch.randn_like(x_0)

        # Add noise according to schedule
        x_t = sqrt_alphas_cumprod.view(-1, 1) * x_0 + sqrt_one_minus_alphas_cumprod.view(-1, 1) * noise

        return x_t, noise

    def predict_noise(self, x_t, t, condition=None):
        """Predict noise from noisy embeddings

        Args:
            x_t: Noisy embeddings
            t: Timestep
            condition: Optional conditioning (e.g., text embedding)
        """
        batch_size = x_t.size(0)
        device = x_t.device

        # Normalize timestep to [0, 1]
        t_normalized = t.float().view(-1, 1) / self.num_timesteps

        # Embed timestep
        t_emb = self.time_mlp(t_normalized.to(device))  # (batch_size, 1)

        # Concatenate with noisy embedding
        x_in = torch.cat([x_t, t_emb], dim=-1)  # (batch_size, embed_dim + 1)

        # Predict noise
        noise_pred = self.noise_predictor(x_in)

        # If condition is provided, guide the noise prediction towards the condition
        if condition is not None:
            # Guidance: pull the denoised result towards the condition
            # This is a simplified form of classifier-free guidance
            guidance_scale = 0.3
            direction = condition - x_t
            noise_pred = noise_pred - guidance_scale * direction

        return noise_pred

    @torch.no_grad()
    def sample(self, batch_size, device, condition=None, num_inference_steps=None):
        """Generate embeddings via reverse diffusion (sampling)

        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            condition: Optional conditioning signal (e.g., text embedding)
            num_inference_steps: Number of denoising steps (None = use all timesteps)
        """
        # Start from random noise, or from condition + noise
        if condition is not None:
            # Start from condition with added noise for diversity
            x_t = condition + torch.randn_like(condition) * 0.5
        else:
            # Start from pure random noise
            x_t = torch.randn(batch_size, self.embed_dim, device=device)

        # Determine which timesteps to use
        if num_inference_steps is None:
            timesteps = list(reversed(range(self.num_timesteps)))
        else:
            # Use a subset of timesteps for faster sampling (DDIM-style)
            step_size = self.num_timesteps // num_inference_steps
            timesteps = list(reversed(range(0, self.num_timesteps, step_size)))

        # Reverse diffusion process
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=device)

            # Predict noise with optional conditioning
            noise_pred = self.predict_noise(x_t, t_batch, condition=condition)

            # Compute coefficients
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # Compute mean
            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            # Predicted x_0
            pred_x_0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

            # Direction pointing to x_t
            dir_x_t = torch.sqrt(1 - alpha_cumprod_prev) * noise_pred

            # Mean
            mean = (torch.sqrt(alpha_cumprod_prev) * pred_x_0 + dir_x_t)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

        return x_t


class MultimodalModel(nn.Module):
    """Multimodal model for image-text matching"""

    def __init__(self, embed_dim=512, use_simple_cnn=True, bert_model_name='bert-base-uncased', freeze_bert=False, cache_dir='./models/bert_cache'):
        super().__init__()

        self.embed_dim = embed_dim

        # Image and text encoders
        self.image_encoder = ImageEncoder(embed_dim, use_simple_cnn=use_simple_cnn)
        self.text_encoder = TextEncoder(embed_dim, bert_model_name=bert_model_name, freeze_bert=freeze_bert, cache_dir=cache_dir)

        # Temperature parameter for contrastive learning (higher for easier training)
        self.temperature = nn.Parameter(torch.ones([]) * 0.1)

    def forward(self, images, input_ids, attention_mask):
        # Encode images and text
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

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


class GenerativeMultimodalModel(nn.Module):
    """Generative multimodal model with encoders, decoders, and diffusion"""

    def __init__(self, vocab_size, embed_dim=512, use_simple_cnn=True,
                 image_size=224, max_text_length=77, num_diffusion_steps=1000,
                 bert_model_name='bert-base-uncased', freeze_bert=False, cache_dir='./models/bert_cache'):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Encoders
        self.image_encoder = ImageEncoder(embed_dim, use_simple_cnn=use_simple_cnn)
        self.text_encoder = TextEncoder(embed_dim, bert_model_name=bert_model_name, freeze_bert=freeze_bert, cache_dir=cache_dir)

        # Decoders
        self.image_decoder = ImageDecoder(embed_dim, image_size)
        self.text_decoder = TextDecoder(vocab_size, embed_dim, max_length=max_text_length)

        # Diffusion model
        self.diffusion = DiffusionModel(embed_dim, num_timesteps=num_diffusion_steps)

        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.1)

    def forward(self, images, input_ids, attention_mask):
        """Forward pass for training"""
        # Encode images and text to unified embedding space
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        # L2 normalize
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, image_features_norm, text_features_norm

    def compute_contrastive_loss(self, image_features, text_features):
        """Compute contrastive learning loss"""
        # Cosine similarity as logits
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric cross-entropy loss
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)

        loss = (loss_i2t + loss_t2i) / 2

        return loss, logits

    def compute_reconstruction_loss(self, images, tokens, image_features, text_features):
        """Compute reconstruction loss for decoders"""
        # Reconstruct images from image features
        reconstructed_images = self.image_decoder(image_features)

        # Normalize original images to [-1, 1] to match decoder output
        images_normalized = images * 2.0 - 1.0
        image_recon_loss = nn.MSELoss()(reconstructed_images, images_normalized)

        # Reconstruct text from text features
        reconstructed_text_logits = self.text_decoder(text_features, target_tokens=tokens, teacher_forcing_ratio=0.5)

        # Reshape for cross entropy
        reconstructed_text_logits = reconstructed_text_logits.view(-1, self.vocab_size)
        tokens_flat = tokens.view(-1)

        text_recon_loss = nn.CrossEntropyLoss(ignore_index=0)(reconstructed_text_logits, tokens_flat)

        return image_recon_loss, text_recon_loss

    def compute_diffusion_loss(self, embeddings):
        """Compute diffusion loss on embeddings"""
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)

        # Forward diffusion (add noise)
        noisy_embeddings, noise = self.diffusion.forward_diffusion(embeddings, t)

        # Predict noise
        predicted_noise = self.diffusion.predict_noise(noisy_embeddings, t)

        # MSE loss between predicted and actual noise
        diffusion_loss = nn.MSELoss()(predicted_noise, noise)

        return diffusion_loss

    def compute_total_loss(self, images, input_ids, attention_mask, tokens,
                          contrastive_weight=1.0,
                          recon_weight=0.5,
                          diffusion_weight=0.3):
        """Compute combined loss for training"""
        # Forward pass
        image_features, text_features, image_features_norm, text_features_norm = self.forward(
            images, input_ids, attention_mask
        )

        # Contrastive loss
        contrastive_loss, logits = self.compute_contrastive_loss(image_features_norm, text_features_norm)

        # Reconstruction loss
        # Use tokens for text decoder (which expects token IDs)
        image_recon_loss, text_recon_loss = self.compute_reconstruction_loss(
            images, tokens, image_features, text_features
        )
        recon_loss = (image_recon_loss + text_recon_loss) / 2

        # Diffusion loss (on both image and text embeddings)
        combined_embeddings = torch.cat([image_features, text_features], dim=0)
        diffusion_loss = self.compute_diffusion_loss(combined_embeddings)

        # Total loss
        total_loss = (
            contrastive_weight * contrastive_loss +
            recon_weight * recon_loss +
            diffusion_weight * diffusion_loss
        )

        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'recon_loss': recon_loss,
            'image_recon_loss': image_recon_loss,
            'text_recon_loss': text_recon_loss,
            'diffusion_loss': diffusion_loss,
            'logits': logits
        }

    @torch.no_grad()
    def generate_from_diffusion(self, num_samples, device, mode='both'):
        """Generate new samples via diffusion

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            mode: 'image', 'text', or 'both'
        """
        # Sample embeddings from diffusion model
        generated_embeddings = self.diffusion.sample(num_samples, device)

        outputs = {}

        if mode in ['image', 'both']:
            # Decode to images
            generated_images = self.image_decoder(generated_embeddings)
            # Convert from [-1, 1] to [0, 1]
            generated_images = (generated_images + 1.0) / 2.0
            outputs['images'] = generated_images

        if mode in ['text', 'both']:
            # Decode to text
            generated_text_logits = self.text_decoder(generated_embeddings)
            generated_tokens = generated_text_logits.argmax(dim=-1)
            outputs['tokens'] = generated_tokens

        return outputs

    @torch.no_grad()
    def cross_modal_generation(self, source, source_type='image'):
        """Generate one modality from another

        Args:
            source: Input data (images or tokens)
            source_type: 'image' or 'text'
        """
        if source_type == 'image':
            # Encode image
            embeddings = self.image_encoder(source)
            # Decode to text
            text_logits = self.text_decoder(embeddings)
            tokens = text_logits.argmax(dim=-1)
            return tokens
        elif source_type == 'text':
            # Encode text
            embeddings = self.text_encoder(source)
            # Decode to image
            images = self.image_decoder(embeddings)
            # Convert from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            return images
        else:
            raise ValueError(f"Unknown source_type: {source_type}")


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

        # Initialize BERT tokenizer
        logger.info("Initializing BERT tokenizer...")
        bert_model_name = config.get('bert_model_name', 'bert-base-uncased')
        cache_dir = config.get('cache_dir', './models/bert_cache')
        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        self.tokenizer.model_max_length = config.get('max_text_length', 77)
        logger.info(f"BERT tokenizer loaded (vocab size: {self.tokenizer.vocab_size})")

        # Load training dataset with tokenizer
        self.train_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config.get('image_size', 224),
            tokenizer=self.tokenizer,
            split='train',
            train_ratio=config.get('train_ratio', 0.9),
            random_seed=config.get('random_seed', 42)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=False  # Disable for MPS
        )

        # Load validation dataset
        self.val_dataset = Flickr30kDataset(
            config['data_dir'],
            image_size=config.get('image_size', 224),
            tokenizer=self.tokenizer,
            split='val',
            train_ratio=config.get('train_ratio', 0.9),
            random_seed=config.get('random_seed', 42)
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=False
        )

        logger.info(f"Training samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")

        # Initialize model (choose between contrastive-only or generative)
        use_generative = config.get('use_generative', False)
        bert_model_name = config.get('bert_model_name', 'bert-base-uncased')
        freeze_bert = config.get('freeze_bert', False)
        cache_dir = config.get('cache_dir', './models/bert_cache')

        if use_generative:
            logger.info("Using GenerativeMultimodalModel (with encoders, decoders, and diffusion)")
            self.model = GenerativeMultimodalModel(
                vocab_size=self.tokenizer.vocab_size,
                embed_dim=config.get('embed_dim', 512),
                use_simple_cnn=config.get('use_simple_cnn', True),
                image_size=config.get('image_size', 224),
                max_text_length=config.get('max_text_length', 77),
                num_diffusion_steps=config.get('num_diffusion_steps', 1000),
                bert_model_name=bert_model_name,
                freeze_bert=freeze_bert,
                cache_dir=cache_dir
            ).to(self.device)
        else:
            logger.info("Using MultimodalModel (contrastive learning only)")
            self.model = MultimodalModel(
                embed_dim=config.get('embed_dim', 512),
                use_simple_cnn=config.get('use_simple_cnn', True),
                bert_model_name=bert_model_name,
                freeze_bert=freeze_bert,
                cache_dir=cache_dir
            ).to(self.device)

        if freeze_bert:
            logger.info("BERT parameters are frozen for faster training")
        else:
            logger.info("BERT parameters will be fine-tuned")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {total_params:,} parameters ({trainable_params:,} trainable)")

        # Initialize optimizer with momentum
        optimizer_name = config.get('optimizer', 'adamw')
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
                weight_decay=config.get('weight_decay', 0.01)
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 0.01),
                nesterov=config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Using {optimizer_name} optimizer with lr={config['learning_rate']}")

        # Learning rate scheduler with warmup
        scheduler_name = config.get('scheduler', 'reduce_on_plateau')
        warmup_epochs = config.get('warmup_epochs', 0)

        if scheduler_name == 'reduce_on_plateau':
            # ReduceLROnPlateau - reduces LR when validation loss plateaus
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.get('lr_decay_factor', 0.5),
                patience=config.get('lr_patience', 3),
                min_lr=config.get('min_lr', 1e-7)
            )
            logger.info(f"Using ReduceLROnPlateau scheduler (patience={config.get('lr_patience', 3)})")
        elif scheduler_name == 'cosine':
            # Cosine Annealing
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=config.get('min_lr', 1e-7)
            )
            logger.info(f"Using CosineAnnealingLR scheduler")
        elif scheduler_name == 'step':
            # Step decay
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('lr_step_size', 10),
                gamma=config.get('lr_decay_factor', 0.5)
            )
            logger.info(f"Using StepLR scheduler (step_size={config.get('lr_step_size', 10)})")
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        # Warmup scheduler wrapper
        if warmup_epochs > 0:
            logger.info(f"Using warmup for {warmup_epochs} epochs")
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )
            self.use_warmup = True
            self.warmup_epochs = warmup_epochs
        else:
            self.use_warmup = False

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

    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0

        use_generative = self.config.get('use_generative', False)

        # Loss accumulators
        if use_generative:
            total_contrastive_loss = 0
            total_recon_loss = 0
            total_diffusion_loss = 0

        # Accuracy accumulators
        total_i2t_top1 = 0
        total_i2t_topk = 0
        total_t2i_top1 = 0
        total_t2i_topk = 0

        pbar = tqdm(self.val_loader, desc='Evaluating', ncols=100)

        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            batch_size = images.size(0)

            # Compute loss based on model type
            if use_generative:
                loss_dict = self.model.compute_total_loss(
                    images, input_ids, attention_mask, tokens,
                    contrastive_weight=self.config.get('contrastive_weight', 1.0),
                    recon_weight=self.config.get('recon_weight', 0.5),
                    diffusion_weight=self.config.get('diffusion_weight', 0.3)
                )
                loss = loss_dict['total_loss']
                logits = loss_dict['logits']

                total_contrastive_loss += loss_dict['contrastive_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_diffusion_loss += loss_dict['diffusion_loss'].item()
            else:
                image_features, text_features = self.model(images, input_ids, attention_mask)
                loss, logits = self.model.compute_loss(image_features, text_features)

            # Compute accuracy
            accuracy = self.compute_accuracy(logits)
            total_i2t_top1 += accuracy['i2t_top1']
            total_i2t_topk += accuracy['i2t_top5']
            total_t2i_top1 += accuracy['t2i_top1']
            total_t2i_topk += accuracy['t2i_top5']

            total_loss += loss.item()
            total_samples += batch_size

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate validation metrics
        num_batches = len(self.val_loader)
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_i2t_top1': total_i2t_top1 / num_batches,
            'val_i2t_top5': total_i2t_topk / num_batches,
            'val_t2i_top1': total_t2i_top1 / num_batches,
            'val_t2i_top5': total_t2i_topk / num_batches,
        }

        if use_generative:
            val_metrics['val_contrastive_loss'] = total_contrastive_loss / num_batches
            val_metrics['val_recon_loss'] = total_recon_loss / num_batches
            val_metrics['val_diffusion_loss'] = total_diffusion_loss / num_batches

        logger.info(f"Validation Results:")
        logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")

        if use_generative:
            logger.info(f"  Contrastive Loss: {val_metrics['val_contrastive_loss']:.4f}")
            logger.info(f"  Reconstruction Loss: {val_metrics['val_recon_loss']:.4f}")
            logger.info(f"  Diffusion Loss: {val_metrics['val_diffusion_loss']:.4f}")

        logger.info(f"  Image->Text: Top-1={val_metrics['val_i2t_top1']*100:.2f}%, Top-5={val_metrics['val_i2t_top5']*100:.2f}%")
        logger.info(f"  Text->Image: Top-1={val_metrics['val_t2i_top1']*100:.2f}%, Top-5={val_metrics['val_t2i_top5']*100:.2f}%")

        return val_metrics

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        epoch_start_time = time.time()

        # Check if using generative model
        use_generative = self.config.get('use_generative', False)

        # Loss accumulators for generative model
        if use_generative:
            total_contrastive_loss = 0
            total_recon_loss = 0
            total_diffusion_loss = 0

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
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            batch_size = images.size(0)

            # Compute loss based on model type
            if use_generative:
                # Generative model with decoders and diffusion
                loss_dict = self.model.compute_total_loss(
                    images, input_ids, attention_mask, tokens,
                    contrastive_weight=self.config.get('contrastive_weight', 1.0),
                    recon_weight=self.config.get('recon_weight', 0.5),
                    diffusion_weight=self.config.get('diffusion_weight', 0.3)
                )
                loss = loss_dict['total_loss']
                logits = loss_dict['logits']

                # Track individual losses
                total_contrastive_loss += loss_dict['contrastive_loss'].item()
                total_recon_loss += loss_dict['recon_loss'].item()
                total_diffusion_loss += loss_dict['diffusion_loss'].item()
            else:
                # Original contrastive model
                image_features, text_features = self.model(images, input_ids, attention_mask)
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
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batches_done:.4f}',
                'i2t': f'{accuracy["i2t_top1"]*100:.1f}%',
                't2i': f'{accuracy["t2i_top1"]*100:.1f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'speed': f'{samples_per_sec:.1f}s/s',
                'ETA': f'{eta_min:02d}:{eta_sec:02d}'
            }

            # Add generative losses to progress bar if applicable
            if use_generative:
                postfix_dict['cont'] = f'{loss_dict["contrastive_loss"].item():.3f}'
                postfix_dict['rec'] = f'{loss_dict["recon_loss"].item():.3f}'
                postfix_dict['diff'] = f'{loss_dict["diffusion_loss"].item():.3f}'

            pbar.set_postfix(postfix_dict)

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
        logger.info(f"  Total Loss: {avg_loss:.4f}")

        if use_generative:
            avg_contrastive = total_contrastive_loss / len(self.train_loader)
            avg_recon = total_recon_loss / len(self.train_loader)
            avg_diffusion = total_diffusion_loss / len(self.train_loader)
            logger.info(f"  Contrastive Loss: {avg_contrastive:.4f}")
            logger.info(f"  Reconstruction Loss: {avg_recon:.4f}")
            logger.info(f"  Diffusion Loss: {avg_diffusion:.4f}")

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
        logger.info(f"Validation steps per epoch: {len(self.val_loader)}")

        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            # Train one epoch
            avg_loss = self.train_epoch(epoch)

            # Evaluate on validation set
            val_metrics = self.evaluate()
            val_loss = val_metrics['val_loss']

            # Update learning rate scheduler
            if self.use_warmup and epoch < self.warmup_epochs:
                # During warmup, use warmup scheduler
                self.warmup_scheduler.step()
                logger.info(f"Warmup phase: epoch {epoch+1}/{self.warmup_epochs}")
            else:
                # After warmup, use main scheduler
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.2e}")

            logger.info(f"Epoch {epoch+1} completed - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model')
                logger.info(f"New best model saved with val loss: {val_loss:.4f}")

            # Save checkpoint per epoch
            self.save_checkpoint(f'epoch-{epoch+1}')

        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

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
            'config': self.config
        }, checkpoint_dir / 'pytorch_model.bin')

        # Save BERT tokenizer separately
        self.tokenizer.save_pretrained(checkpoint_dir / 'tokenizer')

        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main function"""
    config = {
        # Data
        'data_dir': 'data/flickr30k',
        'output_dir': 'outputs',

        # Model
        'vocab_size': 10000,
        'embed_dim': 256,
        'image_size': 224,
        'max_text_length': 77,
        'use_simple_cnn': True,

        # BERT
        'bert_model_name': 'bert-base-uncased',
        'cache_dir': './models/bert_cache',
        'freeze_bert': False,

        # Generative model
        'use_generative': True,
        'num_diffusion_steps': 1000,
        'contrastive_weight': 1.0,
        'recon_weight': 0.5,
        'diffusion_weight': 0.3,

        # Dataset split
        'train_ratio': 0.9,
        'random_seed': 42,

        # Training
        'num_epochs': 20,
        'batch_size': 32,
        'num_workers': 4,

        # Optimizer
        'optimizer': 'adamw',
        'learning_rate': 1e-4,  # Lowered from 5e-4 for stability
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'momentum': 0.9,
        'nesterov': True,

        # Learning rate scheduler
        'scheduler': 'reduce_on_plateau',
        'warmup_epochs': 0,  # Disabled warmup to prevent LR instability
        'lr_patience': 3,
        'lr_decay_factor': 0.5,
        'lr_step_size': 10,
        'min_lr': 1e-7,

        # Save
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

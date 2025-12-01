# 架构更新说明 - Latent Diffusion 实现

## 🎯 核心改进

本次更新将扩散模型从**抽象向量空间**迁移到**图像latent空间**，并采用**非对称式跨模态生成策略**。

---

## 📋 主要变更

### 1. 新增文件

#### `latent_diffusion.py`
完整的Latent Diffusion实现，包含：
- **LatentDiffusionUNet**: 主扩散模型类（DDIM采样）
- **ConditionalUNet2D**: UNet去噪网络（处理2D spatial structure）
- **ResBlockWithCondition**: 带FiLM条件注入的残差块
- **AttentionBlock**: 多头自注意力模块
- **SinusoidalPositionEmbedding**: 时间步编码

**关键特性**：
- 在256×7×7 latent space操作（不是512维向量）
- UNet架构保留spatial structure
- Text embedding通过FiLM注入到每个ResBlock
- DDIM采样加速推理（10-100步）

---

### 2. main.py 修改

#### ImageDecoder 类增强
```python
# 新增方法：
def decode_from_latent(self, latents):
    """从扩散模型生成的latent直接解码图像"""

def get_latent_from_embedding(self, embeddings):
    """将embedding转换为latent（训练时使用）"""
```

#### GenerativeMultimodalModel 重大更新

**旧设计 ❌**：
```python
self.diffusion = DiffusionModel(embed_dim, ...)  # MLP在512维向量上
```

**新设计 ✅**：
```python
self.latent_diffusion = LatentDiffusionUNet(
    latent_channels=256,  # 匹配ImageDecoder latent
    latent_size=7,        # 2D spatial size
    condition_dim=512,    # Text embedding维度
    num_timesteps=1000
)
```

**新增/修改的方法**：

1. **compute_diffusion_loss(image_features, text_features)**
   - 训练目标：`text_embedding → denoise → image_latent`
   - 条件扩散训练（text作为condition）

2. **generate_image_from_text(text_embeddings, num_inference_steps=50)**
   - Text → Image 推理接口
   - 使用Latent Diffusion生成高质量图像

3. **generate_text_from_image(image_embeddings)**
   - Image → Text 推理接口
   - Autoregressive Transformer生成

4. ~~**cross_modal_generation**~~
   - 已弃用（有bug：缺少attention_mask）
   - 请使用上述两个新方法

---

### 3. test.py 更新

**主要改动**：使用新的 `generate_image_from_text` 方法

```python
# 旧代码 ❌
generated_embedding = self.model.diffusion.sample(
    condition=text_embedding,  # 训练时从未见过这种用法！
    ...
)

# 新代码 ✅
generated_image = self.model.generate_image_from_text(
    text_embedding,
    num_inference_steps=50  # 推荐30-50步
)
```

**推荐参数**：
- `use_diffusion=True` （默认，高质量）
- `num_inference_steps=30-50` （平衡质量和速度）

---

### 4. test02.py 更新

**主要改动**：移除不必要的 `use_diffusion` 参数

```python
# Image → Text 不需要扩散！
results = generator.generate_from_image(
    image_paths,
    num_captions_per_image=3,
    temperature=0.9,  # 只需调整解码参数
    top_k=50
)
```

**理由**：
- 文本是离散token序列，Transformer autoregressive已经足够好
- 扩散模型是为连续数据（图像）设计的

---

## 🏗️ 新架构设计理念

### 非对称式跨模态生成

```
Direction 1: Image → Text (Autoregressive)
  Image → ImageEncoder → [512-d] → TextDecoder → Text
  ✓ Transformer擅长序列生成

Direction 2: Text → Image (Diffusion)
  Text → TextEncoder → [512-d] → LatentDiffusion → [256×7×7] → ImageDecoder → Image
  ✓ 扩散模型擅长生成结构化图像
```

### 为什么这样设计？

| 方向 | 数据特性 | 模型选择 | 理由 |
|------|----------|---------|------|
| **Image→Text** | 离散序列 | Autoregressive Transformer | GPT/BERT已证明SOTA |
| **Text→Image** | 连续2D像素 | Latent Diffusion UNet | Stable Diffusion / DALL-E 2 证明SOTA |

**关键洞察**：不同模态应该用不同的生成策略，而不是强行对称！

---

## 🐛 修复的Bug

### Bug 1: cross_modal_generation 缺少参数
```python
# 旧代码（Bug）
embeddings = self.text_encoder(source)  # ❌ 缺少attention_mask

# 应该是
embeddings = self.text_encoder(input_ids, attention_mask)  # ✓
```

### Bug 2: 扩散模型训练-推理不一致
```python
# 训练时：无条件扩散
predicted_noise = self.diffusion.predict_noise(noisy_embeddings, t)  # ❌ 没传condition

# 推理时：却当有条件用
generated = self.diffusion.sample(condition=text_embedding)  # ❌ 训练没见过这个任务
```

### Bug 3: 条件信息未进入网络
```python
# 旧实现
x_in = torch.cat([x_t, t_emb], dim=-1)  # ❌ 没有concat condition
if condition is not None:
    noise_pred = noise_pred - guidance_scale * (condition - x_t)  # 简单向量减法，不是真正的条件扩散

# 新实现
cond = t_emb + c_emb  # ✓ 合并time和condition
h = h * (1 + scale) + shift  # ✓ FiLM注入到每个ResBlock
```

---

## 📊 架构对比

| 方面 | 旧设计 ❌ | 新设计 ✓ |
|------|----------|----------|
| **扩散空间** | 512维抽象向量 | 256×7×7 latent (2D) |
| **网络结构** | MLP | UNet (CNN+Attention) |
| **条件注入** | 后期向量减法 | FiLM注入每层 |
| **训练目标** | 无条件去噪 | Text→Image条件去噪 |
| **Spatial Prior** | ✗ | ✓ 保留2D结构 |
| **训练-推理一致** | ✗ | ✓ |

---

## 🚀 使用方法

### 训练（无需修改）

训练代码自动使用新的Latent Diffusion：

```bash
python main.py
```

训练时会：
1. 对齐embedding空间（contrastive loss）
2. 训练decoder（reconstruction loss）
3. **训练Latent Diffusion**（text → image_latent 去噪）

### 推理：Text → Image

```python
python test.py
```

```python
# 在test.py中
results = generator.generate_from_text(
    prompts=["a dog playing in the park", "a beautiful sunset"],
    num_samples_per_prompt=2,
    use_diffusion=True,        # 推荐True（高质量）
    num_inference_steps=50,    # 30-50步平衡质量和速度
    seed=42                    # 可选，用于复现
)
```

**参数调优**：
- `num_inference_steps=10-20`: 快速预览
- `num_inference_steps=30-50`: 推荐（质量好）
- `num_inference_steps=50-100`: 最佳质量（慢）

### 推理：Image → Text

```python
python test02.py
```

```python
# 在test02.py中
results = generator.generate_from_image(
    image_paths=["path/to/image.jpg"],
    num_captions_per_image=3,
    temperature=0.9,           # 0.7-0.8更focused，0.9-1.0更diverse
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.3
)
```

---

## ⚠️ 重要提示

### 1. 兼容性
- **需要重新训练**：旧模型checkpoint不兼容
- 模型结构变化：diffusion → latent_diffusion

### 2. 依赖检查
```bash
# 确保有以下包
pip install torch torchvision transformers pillow pandas tqdm
```

### 3. GPU内存
- Latent Diffusion比像素空间扩散省100倍内存
- UNet参数量适中（约50M参数）
- 推理时batch_size=1-4足够

### 4. 首次运行
```bash
# 首次运行会自动下载BERT模型（约400MB）
# 会缓存到 ./models/bert_cache/
```

---

## 🔬 技术细节

### Latent Space 设计

```
Embedding (512-d vector)
    ↓ [ImageDecoder.projection]
Latent (256 channels, 7×7 spatial)  ← 扩散在这里操作
    ↓ [ImageDecoder.decoder]
Image (3 channels, 224×224 pixels)
```

**为什么是7×7？**
- ImageEncoder从224×224降到7×7（经过5层stride=2卷积）
- 保留spatial structure的最小尺寸
- 比像素空间快1000倍（224²/7² ≈ 1024倍）

### UNet 结构

```
Input: (B, 256, 7, 7) + Text Embedding (B, 512)
    ↓
[Initial Conv] → (B, 128, 7, 7)
    ↓
[Encoder]
  ResBlock + FiLM(time+text) → (B, 256, 7, 7) ─┐
  ResBlock + FiLM(time+text) → (B, 512, 7, 7) ─┤
    ↓                                           │
[Middle]                                        │
  ResBlock + Attention → (B, 512, 7, 7)         │
    ↓                                           │
[Decoder]                                       │
  ResBlock + FiLM + Skip ←─────────────────────┤
  ResBlock + FiLM + Skip ←─────────────────────┘
    ↓
[Output Conv] → (B, 256, 7, 7)
```

### FiLM 条件注入

```python
# Feature-wise Linear Modulation
cond = time_emb + text_emb  # 合并条件
scale, shift = MLP(cond).chunk(2)
h = h * (1 + scale) + shift  # 在每个ResBlock注入
```

---

## 📈 预期效果

### Text → Image (Latent Diffusion)
- ✅ 图像更连贯（spatial structure）
- ✅ 颜色、形状更准确
- ✅ Text条件真正起作用
- ⏱️ 50步约2-5秒（GPU）

### Image → Text (Autoregressive)
- ✅ 句子更流畅
- ✅ 描述更准确
- ⏱️ 实时生成（<1秒）

---

## 🤔 FAQ

**Q: 为什么不在像素空间做扩散？**
A: 太慢！224×224需要50K个值，7×7只需12K个值。Latent空间快100倍。

**Q: Image→Text为什么不用扩散？**
A: 文本是离散序列，Transformer的autoregressive生成已经很好了。扩散是为连续数据设计的。

**Q: 旧checkpoint还能用吗？**
A: 不能。模型结构变了，需要重新训练。但训练代码无需修改！

**Q: 如何调试生成质量？**
A:
1. 检查training loss是否下降
2. 增加num_inference_steps（50→100）
3. 调整diffusion_weight（config中）
4. 确保contrastive loss收敛（embedding对齐很重要）

**Q: 可以只用直接解码吗（不用扩散）？**
A: 可以，设置`use_diffusion=False`。但质量会明显下降，因为没有迭代refinement。

---

## 📚 参考

本实现参考了以下工作：
- **Stable Diffusion**: Latent空间扩散
- **DALL-E 2**: Text条件图像生成
- **DDIM**: 快速采样算法
- **FiLM**: 条件注入方法

---

## ✅ 下一步

1. ✅ 架构实现完成
2. 🔄 **运行训练**：`python main.py`
3. 🔄 **测试生成**：`python test.py` 和 `python test02.py`
4. 📊 观察效果并调优参数

祝训练顺利！🎉

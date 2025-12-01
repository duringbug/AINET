# 图生文 & 文生图 使用指南

本指南说明如何训练和使用多模态生成模型实现**图生文(Image-to-Text)**和**文生图(Text-to-Image)**功能。

---

## 🎯 功能概述

本项目实现了一个统一的多模态生成模型,支持:

1. **图生文 (Image-to-Text)**: 输入图像 → 生成描述文本
2. **文生图 (Text-to-Image)**: 输入文本 → 生成图像

模型架构:
- **编码器**: 图像编码器(SimpleCNN) + 文本编码器(BERT)
- **解码器**: 图像解码器(Transposed CNN) + 文本解码器(Transformer)
- **生成模块**: Latent Diffusion UNet (用于高质量文生图)

---

## 📋 关键修改说明

### ❌ 之前的问题
```python
'freeze_bert': True,  # BERT被冻结,无法学习
'learning_rate': 1e-3,  # 学习率过高
'diffusion_weight': 0.3,  # 扩散loss权重过大
```
**导致**: Loss卡在12左右,准确率只有12.5%

### ✅ 修复后的配置
```python
'freeze_bert': False,       # 解冻BERT,允许学习
'learning_rate': 2e-4,      # 降低学习率,适合BERT微调
'warmup_epochs': 2,         # 添加warmup
'contrastive_weight': 1.0,  # 对比学习权重
'recon_weight': 0.5,        # 重建权重
'diffusion_weight': 0.1,    # 降低扩散权重
```
**预期效果**:
- Epoch 1-3: 准确率快速上升到 30-50%
- Epoch 5-10: 准确率达到 70-90%
- Epoch 10-15: 准确率接近 100%

---

## 🚀 快速开始

### 1. 训练模型

```bash
# 确保数据集已下载
ls data/coco/coco2017/

# 开始训练(使用修改后的配置)
python main.py
```

**训练监控指标**:
- `contrastive_loss`: 应该从 ~6 降到 ~2
- `i2t` (Image→Text准确率): 应该快速上升
- `t2i` (Text→Image准确率): 应该快速上升
- **如果准确率卡在12%左右,说明BERT被冻结了!**

**预期训练时间** (NVIDIA GPU):
- 每个epoch: ~30-40分钟 (COCO 532K样本)
- 总训练时间: ~8-10小时 (15 epochs)

### 2. 测试生成效果

训练完成后,运行快速测试:

```bash
# 自动测试图生文和文生图
python test_generation.py
```

这会:
1. 自动找到最新的checkpoint
2. 生成5张文生图测试图片
3. 对验证集中的图片生成描述

### 3. 手动推理

#### 文生图 (Text-to-Image)

```bash
python inference.py \
    --checkpoint outputs/best_model \
    --mode t2i \
    --text "a dog playing in the park" \
    --output generated_dog.png \
    --steps 50
```

**参数说明**:
- `--steps`: 扩散步数
  - 10-20 steps: 快速生成,质量一般
  - 30-50 steps: 推荐,质量好
  - 50-100 steps: 最佳质量,但很慢

#### 图生文 (Image-to-Text)

```bash
python inference.py \
    --checkpoint outputs/best_model \
    --mode i2t \
    --image path/to/your/image.jpg
```

#### 同时测试两个功能

```bash
python inference.py \
    --checkpoint outputs/best_model \
    --mode both \
    --image path/to/image.jpg \
    --text "a beautiful landscape" \
    --output generated.png
```

---

## 📊 训练效果预期

### 阶段1: Epoch 1-2 (Warmup)
```
Loss: 8-10
i2t准确率: 20-30%
t2i准确率: 20-30%
```

### 阶段2: Epoch 3-7 (快速学习)
```
Loss: 4-6
i2t准确率: 50-70%
t2i准确率: 50-70%
```

### 阶段3: Epoch 8-15 (收敛)
```
Loss: 2-3
i2t准确率: 85-100%
t2i准确率: 85-100%
```

---

## 🔧 故障排查

### 问题1: Loss不下降,卡在12左右

**原因**: BERT被冻结了

**解决**:
```python
# 检查 main.py 第1640行
'freeze_bert': False,  # 必须是False!
```

### 问题2: Loss下降很慢

**可能原因**:
1. 学习率太高
2. Diffusion loss主导总loss

**解决**:
```python
'learning_rate': 1e-4,  # 降低学习率
'diffusion_weight': 0.05,  # 进一步降低扩散权重
```

### 问题3: 内存不足 (OOM)

**解决**:
```python
'batch_size': 32,  # 从64降到32
'num_workers': 4,  # 从8降到4
```

### 问题4: 生成的图像质量差

**可能原因**:
1. 训练还未收敛
2. 扩散步数太少

**解决**:
1. 多训练几个epoch
2. 推理时使用更多steps: `--steps 100`

---

## 💡 高级用法

### Python API使用

```python
from inference import ImageTextGenerator

# 初始化
generator = ImageTextGenerator('outputs/best_model', device='cuda')

# 文生图
image = generator.text_to_image(
    "a beautiful sunset over mountains",
    num_inference_steps=50,
    save_path='sunset.png'
)

# 图生文
caption = generator.image_to_text('path/to/image.jpg')
print(f"Caption: {caption}")

# 批量生成
texts = ["a cat", "a dog", "a bird"]
images = generator.batch_text_to_image(texts, save_dir='batch_output')
```

### 批量处理

```python
# 批量图生文
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
captions = generator.batch_image_to_text(image_paths)

for img_path, caption in zip(image_paths, captions):
    print(f"{img_path}: {caption}")
```

---

## 📈 性能优化建议

### 加速训练
1. 使用更大的batch size (如果GPU内存足够)
2. 使用混合精度训练 (添加 `torch.cuda.amp`)
3. 减少diffusion步数到500

### 提升生成质量
1. 训练更多epoch (20-30)
2. 使用ResNet50替代SimpleCNN:
   ```python
   'use_simple_cnn': False,
   ```
3. 增加embedding维度:
   ```python
   'embed_dim': 768,
   ```

---

## 🎨 示例输出

### 文生图示例

**输入文本**: "a dog playing in the park"

**生成图像**: 应该生成一张狗在公园玩耍的图片

### 图生文示例

**输入图像**: COCO验证集图片

**生成描述**: "a person riding a bicycle on the street"

---

## 📝 模型架构说明

```
输入图像 → Image Encoder → Image Embedding (512-dim)
                                 ↓
                          [统一向量空间]
                                 ↓
输入文本 → Text Encoder (BERT) → Text Embedding (512-dim)

生成路径:
- 图生文: Image Embedding → Text Decoder → 文本
- 文生图: Text Embedding → Latent Diffusion → Image Decoder → 图像
```

**关键组件**:
1. **Contrastive Learning**: 对齐图像和文本embedding
2. **Reconstruction**: 训练decoder重建原始数据
3. **Latent Diffusion**: 高质量图像生成

---

## 🔍 调试技巧

### 查看各个loss的值

在训练过程中,观察进度条显示:
```
loss=5.23, cont=2.1, rec=2.8, diff=0.3
```

- `cont` (contrastive): 对比学习loss,应该降到1-3
- `rec` (reconstruction): 重建loss,应该降到2-5
- `diff` (diffusion): 扩散loss,应该降到0.1-0.5

### 检查准确率

```
i2t=45.2%, t2i=43.8%
```

- 初期应该快速上升
- 最终应该接近100%

---

## 🎯 最终目标

训练成功的标志:
- ✅ Validation loss < 3.0
- ✅ i2t准确率 > 90%
- ✅ t2i准确率 > 90%
- ✅ 生成的文本描述准确
- ✅ 生成的图像与文本相关

如果达到以上标准,恭喜!你的模型已经可以用于实际的图生文和文生图任务了!

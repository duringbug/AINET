# AINET - 多模态图文生成模型

基于 BERT 的图文匹配与生成模型，使用扩散模型实现跨模态生成。

## 功能特性

- **BERT 文本编码**: 使用预训练 BERT 进行文本向量化
- **图像编码器**: 简单 CNN 或 ResNet50
- **生成模型**: 支持图像和文本的重建与生成
- **扩散模型**: 在统一向量空间中生成新样本
- **对比学习**: 图文匹配与检索

## 快速开始

### 1. 下载 BERT 模型

```bash
# 使用镜像站点下载 (推荐)
python download_bert.py

# 或指定其他模型
python download_bert.py --model bert-base-chinese
```

### 2. 准备数据集

将 Flickr30k 数据集放在 `data/flickr30k/` 目录下。

### 3. 开始训练

```bash
python main.py
```

## 配置说明

在 `main.py` 的 `config` 字典中修改配置:

```python
# BERT 配置
'bert_model_name': 'bert-base-uncased',  # 模型名称，自动从缓存加载
'freeze_bert': False,  # True=冻结BERT, False=微调

# 模型配置
'embed_dim': 256,
'use_simple_cnn': True,

# 训练配置
'num_epochs': 20,
'batch_size': 32,
'learning_rate': 5e-4,
```

## 模型架构

- **文本编码器**: BERT (768维) → 投影层 → 嵌入空间 (256维)
- **图像编码器**: CNN/ResNet → 嵌入空间 (256维)
- **图像解码器**: 从嵌入空间重建图像
- **文本解码器**: 从嵌入空间生成文本
- **扩散模型**: 在嵌入空间中进行扩散生成

## 性能优化

**显存不足**:
```python
'freeze_bert': True,   # 冻结 BERT 参数
'batch_size': 16,      # 减小 batch size
```

**加速训练**:
```python
'freeze_bert': True,   # 只训练投影层
'use_simple_cnn': True,  # 使用简单 CNN
```

## 文件说明

- `main.py`: 主训练脚本
- `download_bert.py`: BERT 模型下载工具
- `download.py`: Flickr30k 数据集下载
- `requirements.txt`: 依赖包列表

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖:
- torch
- transformers
- torchvision
- pillow
- pandas
- tqdm

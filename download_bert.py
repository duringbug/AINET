#!/usr/bin/env python3
"""
简洁的 BERT 模型下载脚本
支持从 HuggingFace 镜像站下载
"""

import os
import sys

def download_bert(model_name='bert-base-uncased', cache_dir='./models/bert_cache', use_mirror=True):
    """
    下载 BERT 模型和 tokenizer

    Args:
        model_name: BERT 模型名称
        cache_dir: 缓存目录
        use_mirror: 是否使用国内镜像
    """
    # 使用镜像加速下载
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用镜像站点: https://hf-mirror.com")

    os.makedirs(cache_dir, exist_ok=True)

    try:
        from transformers import BertModel, BertTokenizer

        print(f"\n正在下载 BERT 模型: {model_name}")
        print(f"保存路径: {cache_dir}\n")

        # 下载 tokenizer
        print("1. 下载 tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"   ✓ Tokenizer 下载成功 (词汇量: {tokenizer.vocab_size})")

        # 下载模型
        print("\n2. 下载 BERT 模型...")
        model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"   ✓ 模型下载成功")
        print(f"   - 隐藏层维度: {model.config.hidden_size}")
        print(f"   - 层数: {model.config.num_hidden_layers}")

        print(f"\n{'='*60}")
        print("下载成功!")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n故障排除:")
        print("  1. 检查网络连接")
        print("  2. 尝试使用镜像: python download_bert.py --mirror")
        print("  3. 手动下载文件到指定目录")
        return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='下载 BERT 模型')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='BERT 模型名称 (默认: bert-base-uncased)')
    parser.add_argument('--cache-dir', type=str, default='./models/bert_cache',
                        help='缓存目录 (默认: ./models/bert_cache)')
    parser.add_argument('--no-mirror', action='store_true',
                        help='不使用镜像站点')

    args = parser.parse_args()

    print("="*60)
    print("BERT 模型下载工具")
    print("="*60)

    success = download_bert(
        model_name=args.model,
        cache_dir=args.cache_dir,
        use_mirror=not args.no_mirror
    )

    sys.exit(0 if success else 1)

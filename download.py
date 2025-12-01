import os
from pathlib import Path
from dotenv import load_dotenv
import kagglehub
import argparse

# 加载环境变量
load_dotenv()

# 设置 Kaggle 认证
kaggle_token = os.getenv('KAGGLE_API_TOKEN')
if kaggle_token:
    os.environ['KAGGLE_KEY'] = kaggle_token
    print("[OK] Kaggle API token loaded from .env")
else:
    raise ValueError("KAGGLE_API_TOKEN not found in .env file")

# 创建 data 目录
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
print(f"[OK] Data directory: {data_dir}")

# 数据集配置
DATASETS = {
    'flickr30k': {
        'name': 'hsankesara/flickr-image-dataset',
        'size': '~30K images',
        'symlink': 'flickr30k'
    },
    'coco': {
        'name': 'awsaf49/coco-2017-dataset',
        'size': '~330K images (train+val)',
        'symlink': 'coco'
    },
    'conceptual_captions': {
        'name': 'google-research-datasets/conceptual-captions',
        'size': '~3.3M images',
        'symlink': 'conceptual_captions'
    }
}

# 参数解析
parser = argparse.ArgumentParser(description='Download image-text datasets')
parser.add_argument('--dataset', type=str, default='coco',
                   choices=list(DATASETS.keys()),
                   help='Dataset to download (default: coco)')
args = parser.parse_args()

dataset_config = DATASETS[args.dataset]
dataset_name = dataset_config['name']

print(f"\n{'='*70}")
print(f"Downloading: {args.dataset.upper()}")
print(f"Dataset: {dataset_name}")
print(f"Size: {dataset_config['size']}")
print(f"{'='*70}\n")
print("This may take a while, please be patient...\n")

try:
    # 使用 kagglehub 下载数据集
    path = kagglehub.dataset_download(dataset_name)

    print(f"\n[OK] Dataset downloaded successfully!")
    print(f"Download location: {path}")
    print(f"\nNote: Data has been downloaded to kagglehub cache directory")
    print(f"You can use this path in your code, or create a symlink to {data_dir}")

    # 创建符号链接到 data 目录
    symlink_path = data_dir / dataset_config['symlink']
    if symlink_path.exists():
        if symlink_path.is_symlink():
            symlink_path.unlink()
        else:
            print(f"[WARNING] {symlink_path} exists and is not a symlink, skipping...")
            symlink_path = None

    if symlink_path:
        symlink_path.symlink_to(path)
        print(f"[OK] Created symlink: {symlink_path} -> {path}")

    # 显示数据集信息
    print(f"\n{'='*70}")
    print(f"Dataset ready!")
    print(f"Path: {symlink_path or path}")
    print(f"{'='*70}\n")

except Exception as e:
    print(f"\n[ERROR] Download failed: {str(e)}")
    print("\nPossible reasons:")
    print("1. Incorrect dataset name")
    print("2. Invalid Kaggle API token")
    print("3. Network connection issues")
    print("4. Dataset not available on Kaggle")
    print(f"\nPlease visit https://www.kaggle.com/datasets and search for {args.dataset}")
    raise

print("\nDone!")
print(f"\nTo use this dataset, update main.py config:")
print(f"  'data_dir': 'data/{dataset_config['symlink']}',")

import os
from pathlib import Path
from dotenv import load_dotenv
import kagglehub

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

# 下载 Flickr30k 数据集
# Flickr30k 在 Kaggle 上的常见数据集路径
dataset_name = "hsankesara/flickr-image-dataset"

print(f"\nStarting download of {dataset_name}...")
print("This may take a while, please be patient...\n")

try:
    # 使用 kagglehub 下载数据集
    # 默认会下载到缓存目录,然后我们可以移动或链接到 data 目录
    path = kagglehub.dataset_download(dataset_name)

    print(f"\n[OK] Dataset downloaded successfully!")
    print(f"Download location: {path}")
    print(f"\nNote: Data has been downloaded to kagglehub cache directory")
    print(f"You can use this path in your code, or create a symlink to {data_dir}")

    # 创建或更新符号链接到 data 目录
    symlink_path = data_dir / "flickr30k"

    # 如果符号链接已存在，检查它是否有效
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink():
            # 如果是符号链接，检查目标是否相同
            if symlink_path.resolve() == Path(path).resolve():
                print(f"[OK] Symlink already exists and points to correct location: {symlink_path} -> {path}")
            else:
                # 更新符号链接到新位置
                symlink_path.unlink()
                symlink_path.symlink_to(path)
                print(f"[OK] Updated symlink: {symlink_path} -> {path}")
        else:
            # 如果是普通目录或文件，警告用户
            print(f"[WARNING] {symlink_path} already exists as a regular file/directory, not creating symlink")
    else:
        # 创建新的符号链接
        symlink_path.symlink_to(path)
        print(f"[OK] Created symlink: {symlink_path} -> {path}")

except Exception as e:
    print(f"\n[ERROR] Download failed: {str(e)}")
    print("\nPossible reasons:")
    print("1. Incorrect dataset name")
    print("2. Invalid Kaggle API token")
    print("3. Network connection issues")
    print("\nPlease visit https://www.kaggle.com/datasets and search for flickr30k to confirm the correct dataset name")
    raise

print("\nDone!")

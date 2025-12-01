import os
from pathlib import Path
from dotenv import load_dotenv
import kagglehub
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 设置 Kaggle 认证
kaggle_token = os.getenv('KAGGLE_API_TOKEN')
if kaggle_token:
    os.environ['KAGGLE_KEY'] = kaggle_token
    logger.info("[OK] Kaggle API token loaded from .env")
else:
    raise ValueError("KAGGLE_API_TOKEN not found in .env file")

# 创建 data 目录
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)
logger.info(f"[OK] Data directory: {data_dir}")

# 可选的风景数据集（选择一个）
datasets = {
    '1': {
        'name': 'arnaud58/landscape-pictures',  # 4000+ landscape images
        'description': 'Landscape Pictures (4000+ images, ~500MB)'
    },
    '2': {
        'name': 'utkarshsaxenadn/landscape-recognition-image-dataset-12k-images',  # 12000+ landscape images
        'description': 'Landscape Recognition Dataset (12000+ images, ~2GB)'
    },
    '3': {
        'name': 'akash2sharma/landscape-images',  # Landscape images
        'description': 'Landscape Images Dataset'
    }
}

print("\n可用的风景数据集:")
print("=" * 70)
for key, dataset in datasets.items():
    print(f"{key}. {dataset['description']}")
    print(f"   Kaggle: {dataset['name']}")
    print()

choice = input("请选择要下载的数据集 (1-3) [默认: 1]: ").strip() or '1'

if choice not in datasets:
    print(f"无效选择，使用默认数据集 1")
    choice = '1'

selected_dataset = datasets[choice]['name']
logger.info(f"\n正在下载数据集: {selected_dataset}")
logger.info("这可能需要一些时间，请耐心等待...\n")

try:
    # 使用 kagglehub 下载数据集
    path = kagglehub.dataset_download(selected_dataset)

    logger.info(f"\n[OK] 数据集下载成功!")
    logger.info(f"下载位置: {path}")

    # 创建或更新符号链接到 data 目录
    symlink_path = data_dir / "landscape"

    # 如果符号链接已存在，检查它是否有效
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink():
            if symlink_path.resolve() == Path(path).resolve():
                logger.info(f"[OK] 符号链接已存在: {symlink_path} -> {path}")
            else:
                symlink_path.unlink()
                symlink_path.symlink_to(path)
                logger.info(f"[OK] 更新符号链接: {symlink_path} -> {path}")
        else:
            logger.warning(f"[WARNING] {symlink_path} 已存在为普通文件/目录，未创建符号链接")
    else:
        symlink_path.symlink_to(path)
        logger.info(f"[OK] 创建符号链接: {symlink_path} -> {path}")

    logger.info(f"\n数据集已准备好!")
    logger.info(f"可以使用以下命令开始训练:")
    logger.info(f"  python ddpm_landscape.py")

except Exception as e:
    logger.error(f"\n[ERROR] 下载失败: {str(e)}")
    logger.error("\n可能的原因:")
    logger.error("1. 数据集名称不正确")
    logger.error("2. Kaggle API token 无效")
    logger.error("3. 网络连接问题")
    logger.error(f"\n你也可以手动下载图片并放在 {data_dir / 'landscape'} 目录中")
    raise

logger.info("\n完成!")

"""墨迹 INKLAB 手写数字识别 - 训练脚本

改编自华为云 ModelArts《GPU迁移到NPU：手写数字识别》示例的训练流程。
- 不依赖 torchvision，自行解析 MNIST IDX 文件（递归查找，兼容 .gz）。
- 标准化：mean=0.1307, std=0.3081。
- 优化器：Adadelta(lr=1.0)；调度器：StepLR(step_size=1, gamma=0.7)。
- 默认：seed=42, epochs=3, CPU 2 线程。

来源：https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html

用法：
    python train.py --data-dir <MNIST 数据目录>
"""
import argparse
import gzip
import json
import logging
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import Net

SOURCE_URL = "https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html"
BASE_DIR = Path(__file__).resolve().parent

MEAN = 0.1307
STD = 0.3081

TRAIN_IMAGE_NAMES = [
    "train-images-idx3-ubyte",
    "train-images.idx3-ubyte",
    "train_images_idx3_ubyte",
]
TRAIN_LABEL_NAMES = [
    "train-labels-idx1-ubyte",
    "train-labels.idx1-ubyte",
    "train_labels_idx1_ubyte",
]
TEST_IMAGE_NAMES = [
    "t10k-images-idx3-ubyte",
    "t10k-images.idx3-ubyte",
    "t10k_images_idx3_ubyte",
]
TEST_LABEL_NAMES = [
    "t10k-labels-idx1-ubyte",
    "t10k-labels.idx1-ubyte",
    "t10k_labels_idx1_ubyte",
]


def find_data_file(data_dir, names):
    data_dir = Path(data_dir)
    patterns = []
    for name in names:
        patterns.append(name)
        patterns.append(name + ".gz")
    for pattern in patterns:
        matches = sorted(data_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def read_idx(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        dtype_code = (magic >> 8) & 0xFF
        ndim = magic & 0xFF
        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64,
        }
        if dtype_code not in dtype_map:
            raise ValueError("不支持的 IDX 数据类型: 0x%02X" % dtype_code)
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=dtype_map[dtype_code])
        data = data.reshape(dims)
    return np.array(data, copy=True)


def load_mnist(data_dir):
    ti = find_data_file(data_dir, TRAIN_IMAGE_NAMES)
    tl = find_data_file(data_dir, TRAIN_LABEL_NAMES)
    ei = find_data_file(data_dir, TEST_IMAGE_NAMES)
    el = find_data_file(data_dir, TEST_LABEL_NAMES)
    missing = []
    if ti is None:
        missing.append("训练图像")
    if tl is None:
        missing.append("训练标签")
    if ei is None:
        missing.append("测试图像")
    if el is None:
        missing.append("测试标签")
    if missing:
        raise FileNotFoundError(
            "在 %s 中未找到 MNIST 文件: %s" % (data_dir, "、".join(missing))
        )
    logging.info("训练图像: %s", ti)
    logging.info("训练标签: %s", tl)
    logging.info("测试图像: %s", ei)
    logging.info("测试标签: %s", el)
    x_train = read_idx(ti)
    y_train = read_idx(tl)
    x_test = read_idx(ei)
    y_test = read_idx(el)
    return x_train, y_train, x_test, y_test


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        images = images.astype(np.float32) / 255.0
        images = (images - MEAN) / STD
        self.images = images
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)
        label = int(self.labels[idx])
        return img, label


def train_one_epoch(model, device, loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    num_samples = 0
    total = len(loader.dataset)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        num_samples += data.size(0)
        if batch_idx % 100 == 0:
            logging.info(
                "Train Epoch: %d [%5d/%5d (%.0f%%)] Loss: %.6f",
                epoch,
                num_samples,
                total,
                100.0 * num_samples / total,
                loss.item(),
            )
    return running_loss / num_samples


def evaluate(model, device, loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    confusion = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            t_np = target.cpu().numpy()
            p_np = pred.cpu().numpy()
            for t, p in zip(t_np, p_np):
                confusion[int(t), int(p)] += 1
    total = len(loader.dataset)
    test_loss /= total
    accuracy = correct / total
    return test_loss, correct, accuracy, confusion


def save_sample_images(x_test, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for img, label in zip(x_test, y_test):
        label = int(label)
        if label not in saved:
            Image.fromarray(img).save(output_dir / ("%d.png" % label))
            saved[label] = True
        if len(saved) == 10:
            break
    return sorted(saved.keys())


def setup_logging(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(description="墨迹 INKLAB MNIST 训练脚本")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="MNIST IDX 数据所在目录（递归查找，兼容 .gz）",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir / "training.log")

    logging.info("=== 墨迹 INKLAB 手写数字识别 - 训练开始 ===")
    logging.info("参数: %s", vars(args))
    logging.info("来源: %s", SOURCE_URL)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    logging.info("设备: %s, 线程数: %d, 随机种子: %d", device, args.threads, args.seed)

    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    logging.info("训练集: %d 样本, 测试集: %d 样本", len(y_train), len(y_test))

    train_dataset = MNISTDataset(x_train, y_train)
    test_dataset = MNISTDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, epoch)
        test_loss, correct, accuracy, confusion = evaluate(model, device, test_loader)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "test_accuracy": float(accuracy),
            "test_correct": int(correct),
            "test_total": len(test_dataset),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        logging.info(
            "Epoch %d 完成: train_loss=%.4f test_loss=%.4f accuracy=%.4f (%d/%d)",
            epoch,
            train_loss,
            test_loss,
            accuracy,
            correct,
            len(test_dataset),
        )
    elapsed = time.time() - start_time

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "mnist.pt"
    torch.save(model.state_dict(), str(model_path))
    logging.info("模型已保存: %s", model_path)

    samples_dir = output_dir / "static" / "samples"
    saved_labels = save_sample_images(x_test, y_test, samples_dir)
    logging.info(
        "测试样本图片已保存: %s (%s)",
        samples_dir,
        ", ".join(str(l) for l in saved_labels),
    )

    final = history[-1]
    metrics = {
        "accuracy": final["test_accuracy"],
        "loss": final["test_loss"],
        "correct": final["test_correct"],
        "total_samples": final["test_total"],
        "train_samples": len(train_dataset),
        "history": history,
        "confusion_matrix": confusion.tolist(),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "source_url": SOURCE_URL,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "Net",
        "model_path": str(model_path),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "seed": args.seed,
            "threads": args.threads,
            "optimizer": "Adadelta",
            "scheduler": "StepLR(step_size=1, gamma=0.7)",
            "normalization": {"mean": MEAN, "std": STD},
        },
    }
    metrics_path = output_dir / "metrics.json"
    with open(str(metrics_path), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logging.info("指标已保存: %s", metrics_path)

    logging.info(
        "=== 训练完成: 耗时 %.1f 秒, 最终准确率 %.4f ===",
        elapsed,
        final["test_accuracy"],
    )


if __name__ == "__main__":
    main()
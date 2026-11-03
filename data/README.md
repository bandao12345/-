# MNIST 原始样本

`MNIST_Data.zip` 是原项目已使用的数据压缩包，包含60,000张训练图片、60,000个训练标签、10,000张测试图片和10,000个测试标签。图片以IDX二进制文件保存，每张28×28像素。数量、解压后大小与SHA256已经核对，详情在 `manifest.json`。

在仓库根目录的 PowerShell 解压：

```powershell
Expand-Archive -LiteralPath .\data\MNIST_Data.zip -DestinationPath .\data\mnist
```

训练脚本会递归查找数据。先运行一次 `legacy/mnist/start.bat` 准备原版环境，再在根目录复现训练：

```powershell
.\legacy\mnist\.venv\Scripts\python.exe -X utf8 legacy\mnist\train.py --data-dir data\mnist --epochs 3 --threads 2 --seed 42 --output-dir training-output
```

`training-output` 与解压出的IDX已被Git忽略，复现训练不覆盖仓库中已有权重。正常演示不需要再次训练。

数据来源为华为原实验采用的公开MNIST。公开下载镜像及校验字段同时记录在清单中，镜像来自PyTorch官方数据集实现的资源地址。这些数字样本仅用于原单数字CNN，不是新版汉字OCR的完整训练集。

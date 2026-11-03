# 原单数字 CNN 项目

本目录保留原项目的网络、训练脚本、Flask应用、网页、已训练权重、真实训练输出和指标。双击本目录 `start.bat`，地址为 http://127.0.0.1:5001/。

原实验来源：[华为ModelArts《GPU迁移到NPU：手写数字识别》](https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html)。实际训练为PyTorch、本机CPU、MNIST训练60,000张、测试10,000张、3轮，正确9892张，测试准确率98.92%。本项目没有实施GPU/NPU迁移。

原权重为 `models/mnist.pt`。数据压缩包和复现步骤见 [../../data/README.md](../../data/README.md)。新版应用位于仓库根目录，端口5003，请从根目录启动新版。

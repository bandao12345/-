# 模型文件

仓库包含固定下载地址、文件大小、SHA256 与上游许可证，不直接上传大体积的 OCR / LLM 权重。

| 用途 | 清单 | 恢复命令 |
| --- | --- | --- |
| 当前 PP-OCRv6 medium | `ocr-v6/manifest.json` | `python download_models.py --version v6` |
| 历史 PP-OCRv5 | `ocr/manifest.json` | `python download_models.py --version v5` |
| 可选本机 Qwen 与 llama.cpp | `llm/manifest.json` | `python tools/download_local_model.py` |

其中 OCR V6 识别、检测及方向模型分别负责读字、定位与方向辅助。正常新版只需要 V6，V5留作历史来源记录。

原单数字 CNN 权重直接保留于 `../legacy/mnist/models/mnist.pt`，约4.8 MB。`mnist_baseline_metrics.json`记录原模型训练及测试成绩，不能作为新版通用OCR成绩。

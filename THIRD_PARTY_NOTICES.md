# 第三方来源与许可

- 原实验：华为ModelArts《GPU迁移到NPU：手写数字识别》，链接见根README。框架选用PyTorch，实际CPU训练。
- MNIST：原实验采用的公开手写数字数据，压缩包、数量与SHA256记录在data目录。PyTorch官方镜像资源出处见data/manifest.json。
- OCR：RapidAI/RapidOCR与PaddlePaddle/PaddleOCR，模型固定地址及SHA256在models/ocr-v6/manifest.json。对应许可证原文保留在模型目录与static/samples/PaddleOCR-LICENSE.txt。
- 网页样例：来源与人工规范笔画/公开裁片性质见static/samples/SOURCES.md及manifest.json。tests中仅保留公开来源的手写裁片，没有复制开发者私人识别数据库或私人笔迹回归样本。
- Qwen：官方Qwen/Qwen2.5-1.5B-Instruct-GGUF，固定revision、哈希与Apache-2.0许可证保留在models/llm。
- llama.cpp：ggml-org/llama.cpp，固定版本、运行时哈希与MIT许可证保留在models/llm。
- MCP：官方modelcontextprotocol/python-sdk，项目适配器通过stdio连接本机服务。

本文件仅说明来源，不代替上游许可证。上游模型、数据与依赖按各自许可使用。

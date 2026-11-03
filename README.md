# 墨迹 INKLAB 手写资料学习助手

工程训练2期末大作业，选择**方案四：基于华为网站实验改造成全栈项目**。

原实验为华为 ModelArts [《GPU迁移到NPU：手写数字识别》](https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html)。本项目使用 PyTorch 完成原单数字 CNN 的 CPU 训练与推理，再扩展为汉字和字母数字识别、人工校对、SQLite 记录管理、学习资料库与引用问答。

本目录从已完成项目中整理，不包含账号密钥、本机私人数据库、虚拟环境或临时演示授权。已有源码与答辩 PPT 均保留。

## 提交内容

| 内容 | 位置 |
| --- | --- |
| 新版全栈项目 | 根目录的 Python 文件、`templates/`、`static/` |
| 答辩 PPT，14页，备注含讲稿与备用问答 | [presentations/](presentations/) |
| 原单数字 CNN、训练脚本、4.8 MB 权重及指标 | [legacy/mnist/](legacy/mnist/) |
| 60,000张训练图与10,000张测试图的原始压缩包 | [data/MNIST_Data.zip](data/MNIST_Data.zip) |
| 数据集数量、大小及 SHA256 | [data/manifest.json](data/manifest.json) |
| 网页演示样本与来源 | [static/samples/](static/samples/) |
| OCR / 千问模型清单与许可证 | [models/](models/) |
| 课程要求对应说明 | [docs/作业要求与项目对应.md](docs/作业要求与项目对应.md) |
| 历史验收摘要 | [docs/verification_3.json](docs/verification_3.json) |

## 功能与技术

- HTML、CSS、JavaScript 和 Canvas 提供六页界面与画板，Flask API 配合 Waitress 承载业务。
- PP-OCRv6 / RapidOCR / ONNX Runtime 在 CPU 上识别文字，同一画板提供“汉字”和“字母／数字”两种模式。
- 原图、模型文字、人工校对、备注分别保存，支持筛选与 CSV 导出。汉字结构复核提供候选，等待用户确认。
- SQLite 分别保存识别历史与学习资料。确认后的 OCR 文本可以入库，也可导入 TXT、Markdown 和含文本 PDF。
- 中文 BM25 词法检索返回原文片段及来源。默认提供原文摘录，启用配置好的模型后执行 RAG 生成。
- 可选本机 Qwen2.5-1.5B-Instruct Q4_K_M，通过 llama.cpp 运行。问答质量受小模型与资料覆盖范围限制。
- 有限规则工具流程展示真实调用轨迹。MCP stdio 提供五个工具，项目 [SKILL.md](SKILL.md) 说明核对、入库及引用步骤。
- 提供 ngrok 独立演示入口、Dockerfile 和 compose 配置。ngrok 已有历史实测，Docker 与云服务器未实际部署。

## 启动新版

建议使用 **Windows 64位 Python 3.12**。首次在仓库根目录打开 PowerShell，分别执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 download_models.py --version v6
.\start.bat
```

浏览器地址为 **http://127.0.0.1:5003/**。以后双击根目录 `start.bat` 即可；服务运行时会复用已有服务。下载完成后，正常 OCR、SQLite 和词法检索可以离线运行。数据库首次启动时自动创建在 `instance/`，仓库没有携带开发者的私人记录。

提交版未直接携带约133 MB的 OCR 权重和约1.04 GiB的千问权重，现有下载脚本按清单恢复并核验 SHA256。**先恢复 V6 权重再启动**，这样可以保持仓库便于上传。[模型说明](models/README.md)

### 可选本机千问

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\download_local_model.py
```

该脚本下载固定版本的千问 GGUF 与 Windows x64 CPU 运行时到项目目录，完成后重新启动 `start.bat`。进入“知识助手”，勾选“使用本机模型生成”。不开启模型时仍可使用识别、资料库和检索摘录。

也可参照 `local_settings.example.json` 在本机配置兼容接口，填写后的 `local_settings.json` 已被 Git 忽略。资料问答需要相关资料；完整聊天历史尚未持久保存，识别记录查询不等于完整聊天回忆。

### 原单数字项目

进入 `legacy/mnist/`，双击该目录的 `start.bat`，使用 **http://127.0.0.1:5001/**。原版会准备独立的 PyTorch CPU 环境。原训练指标为 MNIST 测试集 **9892 / 10000，98.92%**，仅适用于该单数字 CNN。完整训练数据的解压与复现说明见 [data/README.md](data/README.md)。

## MCP 与基础测试

先启动新版，再从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\mcp_smoke.py
```

程序实际完成 MCP 握手、五个工具的发现与资料检索，默认不调用大语言模型。独立基础测试使用临时数据库，不需要下载 OCR 或千问权重：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

原开发环境在2026-10-22完成过131项自动检查、30项网页检查、13项公网HTTP检查和5项公网浏览器检查，记录在历史验收文件中。这些计数不是OCR准确率。本次提交版的49项公共基础测试全部通过；未公开私人笔迹回归图或依赖它们的测试，测试范围与原开发环境有区别。整理核对见 [提交检查记录](docs/submission_check_20261103.json)。

## 外网演示

先让本机新版正常运行。ngrok 程序和账号配置由演示者在本机准备，账号令牌不属于仓库内容。将 ngrok 加入 PATH 后，双击 `start-demo.bat`。该入口使用独立的5005演示数据库，显示本轮完整授权入口。首次遇到ngrok提示页时点击 Visit Site。新启动时保持窗口运行，Ctrl+C结束本次演示。

此地址依赖本机和隧道进程。GitHub托管源码，本项目的Flask服务不能直接通过GitHub Pages运行。

Docker 文件沿用原项目。若自行构建容器，须先在构建目录恢复V6权重，再执行 `docker compose up --build -d`。容器默认不启动Windows本机千问，可另配模型服务；这些容器步骤尚未实际验收。

## 模型、数据和协作来源

原 MNIST CNN 实际完成本机 CPU 训练3轮。新版 OCR 与 Qwen 使用上游预训练权重，未在本项目训练或微调，也未完成GPU/NPU迁移。CodeArts完成原模型、训练脚本与初版服务，后续全栈与OCR升级由Codex协作完善。真实会话未伪造为全部由CodeArts完成，个人开发会话不随公共仓库发布。

上游来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。课程方案及功能对应见 [项目对应说明](docs/作业要求与项目对应.md)。

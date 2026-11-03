# 上传 GitHub 并提交作业

上传的是本文件夹中的内容，不是原工作区整个“墨迹识别实验”文件夹。

1. 在自己的GitHub账号创建仓库，例如 `inklab-engineering-training`。
2. 使用GitHub Desktop把本文件夹加入仓库并发布，或在GitHub网页选 Add file / Upload files，分批拖入文件与目录。README、源码、static、templates、models清单、legacy、data和presentations都应保留各自目录结构。网页可能不显示点开头的文件，GitHub Desktop可一并处理 `.gitignore`、`.gitattributes`。
3. 发布后确认首页能看到项目README，`presentations` 能找到PPT，`legacy/mnist/models` 能找到原CNN权重，`data` 能找到MNIST压缩包。
4. 将仓库网页链接按老师要求提交到智慧树，按课程要求参加方案四答辩。仓库访问权限应确保老师能打开。

本次整理已经排除私有数据库、密钥、虚拟环境和临时授权链接。不要再把原项目的 `instance/`、`.env`、填写后的 `local_settings.json` 或ngrok账号配置拖入仓库。

GitHub网页单文件上传限25MiB，Git普通文件上限100MiB，见[官方上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。当前提交文件均小于25MiB。OCR与千问大权重使用现成下载脚本，避免上传限制；MNIST原始数据已经压缩为约11MB，可随本目录提交。

GitHub保存源码与PPT，不会自动运行Flask。演示时仍需按根README在本机启动应用，再按需开启ngrok。

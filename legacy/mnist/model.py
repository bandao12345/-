"""墨迹 INKLAB 手写数字识别 - CNN 模型定义

网络结构改编自华为云 ModelArts《GPU迁移到NPU：手写数字识别》示例。
来源：https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """MNIST 分类 CNN。

    Conv1->32(3)+ReLU -> Conv32->64(3)+ReLU -> MaxPool2
    -> Dropout0.25 -> Flatten(9216) -> Linear128+ReLU
    -> Dropout0.5 -> Linear10 -> log_softmax
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)